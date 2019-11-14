import argparse
import os
import tqdm

import numpy as np
import librosa
from torch.autograd import Variable
from torch.distributions import Categorical

from utils import audio_to_onehot, load_latest_model_from, DataParallel, mu_law_encoding, mu_law_expanding
from wavenet_model import *
from data import AudioData, AudioDataset
from trainer import *

import torch

import sys
sys.path.append("../")
from sequnet_res import Sequnet

NUM_CLASSES = 256

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--levels', type=int, default=10,
                    help='# of levels (default: 10)')
parser.add_argument('--features', type=int, default=128,
                    help='# of feature channels per layer for Seq-U-Net (default: 128)')
parser.add_argument('--snapshot_dir', type=str,
                    help='Directory from where to read checkpoint file of model')
parser.add_argument('--batch_size', type=int, default=4,
                    help="Batch size")
parser.add_argument('--num_batches', type=int, default=1,
                    help="Number of batches to generate (audio clips=num_batches*batch_size)")
parser.add_argument('--model', type=str, default='wavenet',
                    help='Model to use (wavenet/sequnet)')
parser.add_argument('--depth', type=int, default=1,
                    help="Number of residual blocks in each Seq-U-Net downsampling/upsampling block")
parser.add_argument('--hourglass', type=int, default=1,
                    help="How many hourglasses for Seq-U-Net")
parser.add_argument('--preprocessed_dataset_dir', type=str, default="dataset",
                    help='Dataset path')
parser.add_argument('--fast', type=int, default=1,
                    help='Fast generation by reusing previous convolutional layer activations (default 1)')
parser.add_argument('--gen_samples', type=int, default=16000,
                    help='How many samples of audio should be generated/added (in case of conditional generation)')
parser.add_argument('--temp', type=float, default=1,
                    help='Temperature (default: 1). Low/high temperature means more conservative/chaotic outputs')
parser.add_argument('--conditional', type=int, default=1,
                    help='1 to complete a test audio sample, 0 to generate from scratch (unconditional)')
parser.add_argument('--conditional_duration', type=float, default=1.,
                    help='Number of seconds of context to prepend to the generated audio files.'
                         'Will be set to at least the models receptive field!')
parser.add_argument('--sr', type=int, default=16000, help="Sampling rate")
parser.add_argument('--output_dir', type=str, default="generated_samples", help="Where to save audio")

args = parser.parse_args()

dtype = torch.FloatTensor
ltype = torch.LongTensor

def generate(model,
             classes,
             sample_length,
             device,
             fast,
             batch_size=None,
             first_samples=None,
             temperature=1.0):
    model.eval()

    # Prepare conditional samples given, if any
    crop_front = 0
    if first_samples is None: # No conditioning
        assert(batch_size is not None) # Need batch size to know how many samples to generate!
        # Generate num_samples silence signals to start feeding the model with
        first_samples = torch.zeros((batch_size, model.input_size), dtype=torch.float32)
        crop_front = model.input_size
        print("Unconditional generation of " + str(sample_length) + " samples..")
        mode = "pytorch"
    else:
        if isinstance(first_samples, torch.Tensor):
            mode = "pytorch"
        else:
            assert(isinstance(first_samples, np.ndarray))
            mode = "numpy"
            first_samples = torch.tensor(first_samples)

        print("Completing input audio signal of length " + str(first_samples.shape[1]) + " with " + str(sample_length) + " extra samples...")

    # Now we have first_samples as float or int torch tensor. If float, need to convert to mulaw-int
    assert(first_samples.dim() == 2) # Has to be [batch size, samples]
    if first_samples.dtype == torch.float32:
        # Convert audio signal to mu-law int64 index vector
        first_samples = mu_law_encoding(first_samples, qc=256)
    assert(first_samples.dtype == torch.int64) # Index vectors from here on
    first_samples = first_samples.to(device)

    # Pad conditioning if it is not enough by prepending zeros
    num_pad = model.input_size - first_samples.size(1)
    if num_pad > 0:
        first_samples = torch.cat([mu_law_encoding(torch.zeros((first_samples.shape[0], num_pad),dtype=torch.float32)), first_samples], dim=1)
        crop_front = num_pad

    with torch.no_grad(), tqdm(total=sample_length) as pbar:
        generated = Variable(first_samples)
        model_input = Variable(torch.FloatTensor(generated.shape[0], classes, model.input_size).zero_()).to(device)
        model_input_new = Variable(torch.FloatTensor(generated.shape[0], classes, 1).zero_()).to(device)
        for i in range(sample_length):
            if not fast or i==0: # Set input to model in normal generation mode or in first iteration of fast mode
                model_input.zero_()
                model_input = model_input.scatter_(1, generated[:, -model.input_size:].view(-1, 1, model.input_size), 1.)

            if fast:
                if i == 0:
                    x = model(model_input, mode="save")[:, :, -1] # Save activations on first passthrough
                else:
                    model_input_new.zero_()
                    model_input_new = model_input_new.scatter_(1,generated[:, -1].view(-1, 1, 1), 1.)
                    x = model(model_input_new, mode="step").squeeze(2)
            else:
                x = model(model_input)[:, :, -1]


            if temperature != 1.0:
                x /= temperature
            pdf = Categorical(logits=x)
            x = pdf.sample()

            pbar.update(1)

            generated = torch.cat([generated, x.unsqueeze(1)], 1)

        # Concat first samples with generated ones
        generated = torch.cat([first_samples, generated[:,-sample_length:]], dim=1)
        # Crop front if padded earlier
        if crop_front > 0:
            generated = generated[:,crop_front:]

        # Decode into float from mulaw
        mu_gen = mu_law_expanding(generated, qc=256)

        # If we had numpy input, convert to numpy
        if mode == "numpy":
            mu_gen = mu_gen.cpu().numpy()

    model.train()
    return mu_gen

np.random.seed(1337)
torch.random.manual_seed(1337)

# MODEL
if args.model == "wavenet":
    model = WaveNetModel(layers=13,
                     blocks=4,
                     dilation_channels=128,
                     residual_channels=128,
                     skip_channels=512,
                     output_length=1,
                     dtype=dtype,
                     bias=True)
elif args.model == "sequnet":
    model = Sequnet(NUM_CLASSES, args.features, args.levels, NUM_CLASSES, kernel_size=5, target_output_size=1, depth=args.depth)
else:
    raise NotImplementedError("Could not find model " + str(args.model))

if args.cuda:
    model = DataParallel(model)
    print("move model to gpu")
    model.cuda()
load_latest_model_from(model, None, args.snapshot_dir)

print('model: ', model)
print('input length: ', model.input_size)
print('output length: ', model.output_size)
print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

if args.conditional:
    test_audio_data = AudioData(os.path.join(args.preprocessed_dataset_dir, "audio_test.hdf5"), sr=args.sr, channels=1)
    transform = lambda x : audio_to_onehot(x, model.output_size, NUM_CLASSES)
    data = AudioDataset(test_audio_data, input_size=1,
                        context_front=max(int(args.conditional_duration*args.sr),model.input_size),
                        hop_size=20*args.sr, random_hops=True, audio_transform=transform)

    print('the dataset has ' + str(len(data)) + ' items')

for i in range(args.num_batches):
    if args.conditional:
        idx = np.random.choice(len(data), args.batch_size)
        start_data = [torch.max(data[i][0], 0)[1] for i in idx]
        start_data = torch.stack(start_data)
    else:
        start_data = None

    generated_batch = generate(model, NUM_CLASSES, args.gen_samples, "cuda", args.fast, first_samples=start_data, batch_size=args.batch_size, temperature=args.temp).cpu().numpy()

    print(generated_batch)
    for example_num in range(generated_batch.shape[0]):
        librosa.output.write_wav(os.path.join(args.output_dir, 'clip_' + str(i) + "_" + str(example_num) + '_.wav'), generated_batch[example_num], sr=args.sr)