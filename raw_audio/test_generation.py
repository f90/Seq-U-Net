'''
Script that makes sure that the fast generation procedure produces the same results as the slow one
'''

import argparse

from utils import audio_to_onehot, load_latest_model_from, DataParallel, mu_law_encoding, mu_law_expanding
from trainer import *

import sys

from wavenet_model import WaveNetModel

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
parser.add_argument('--depth', type=int, default=1,
                    help="Number of residual blocks in each Seq-U-Net downsampling/upsampling block")
parser.add_argument('--snapshot_dir', type=str, default=None,
                    help='Directory from where to read checkpoint file of model')
parser.add_argument('--batch_size', type=int, default=16,
                    help="Batch size")
parser.add_argument('--dataset_dir', type=str, default="dataset",
                    help='Dataset path')
parser.add_argument('--model', type=str, default='wavenet',
                    help='Model to use (wavenet/sequnet)')
args = parser.parse_args()

dtype = torch.FloatTensor
ltype = torch.LongTensor

# MODEL
if args.model == "wavenet":
    model = WaveNetModel(layers=13,
                     blocks=4,
                     dilation_channels=128,
                     residual_channels=128,
                     skip_channels=512,
                     output_length=2,
                     dtype=dtype,
                     bias=True)
    model.cycles = 1
elif args.model == "sequnet":
    model = Sequnet(NUM_CLASSES, args.features, args.levels, NUM_CLASSES, kernel_size=5, target_output_size=2**args.levels, depth=args.depth)

if args.cuda:
    model = DataParallel(model)
    print("move model to gpu")
    model.cuda()
if args.snapshot_dir is not None:
    load_latest_model_from(model, None, args.snapshot_dir)

print('model: ', model)
print('input length: ', model.input_size)
print('output length: ', model.output_size)
print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

model.eval()

with torch.no_grad(), tqdm(total=model.cycles) as pbar:
    input = torch.ones((1, 256, model.input_size))
    if args.cuda:
        input = input.cuda()

    normal_out = model(input)
    save_out = model(input, mode="save")

    assert(torch.allclose(normal_out, save_out, atol=1e-5)) # Normal output should be the same no matter whether we save or not
    assert(torch.allclose(normal_out[:,:,-model.cycles:], normal_out[:,:,-2*model.cycles:-model.cycles], atol=1e-5)) # Model with C cycles should repeat its output

    for i in range(model.cycles):
        x = model(input[:, :, -1:], mode="step").squeeze(2)
        assert(torch.allclose(x, normal_out[:,:,-(model.cycles-i)], atol=1e-5)) # Output from step mode should be the same as in normal mode
        pbar.update(1)

print("TEST SUCCESSFULLY FINISHED")