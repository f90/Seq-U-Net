import argparse
import glob
import os

from data import AudioData, AudioDataset
from utils import audio_to_onehot
from wavenet_model import *
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

import sys

from utils import DataParallel

sys.path.append("../")
from sequnet_res import Sequnet

dtype = torch.FloatTensor
ltype = torch.LongTensor

## TRAIN PARAMETERS
parser = argparse.ArgumentParser(description='Sequence Modeling - Raw Audio')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of data loader worker threads (default: 1)')
parser.add_argument('--levels', type=int, default=10,
                    help='# of levels (default: 10)')
parser.add_argument('--features', type=int, default=128,
                    help='# of feature channels per layer for Seq-U-Net (default: 128)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--log_dir', type=str, default='logs/wavenet',
                    help='Folder to write logs into')
parser.add_argument('--dataset_dir', type=str, default="dataset",
                    help='Dataset path')
parser.add_argument('--preprocessed_dataset_dir', type=str, default="dataset",
                    help='Path where preprocessed dataset is saved/loaded')
parser.add_argument('--snapshot_dir', type=str, default='snapshots/wavenet',
                    help='Folder to write checkpoints into')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--batch_size', type=int, default=16,
                    help="Batch size")
parser.add_argument('--hourglass', type=int, default=1,
                    help="How many hourglasses for Seq-U-Net")
parser.add_argument('--output_size', type=int, default=5000,
                    help="Number of target audio samples that should be predicted for each example in the batch")
parser.add_argument('--depth', type=int, default=1,
                    help="Number of residual blocks in each Seq-U-Net downsampling/upsampling block")
parser.add_argument('--eps', type=float, default=1e-8,
                    help="Adam Epsilon value for training stabilisation")
parser.add_argument('--model', type=str, default='wavenet',
                    help='Model to use (wavenet/sequnet)')
parser.add_argument('--sr', type=int, default=16000,
                    help="Sampling rate")

args = parser.parse_args()

# Check for CUDA
if args.cuda:
    if not torch.cuda.is_available():
        raise EnvironmentError("Requested CUDA operation, but no CUDA device found!")
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    print('use cpu')
    if torch.cuda.is_available():
        print("WARNING: You chose to run CPU mode, but have a CUDA device. You probably want to run with --cuda")
    dtype = torch.FloatTensor
    ltype = torch.LongTensor

torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5

# MODEL
NUM_CLASSES = 256
if args.model == "wavenet":
    model = WaveNetModel(layers=13,
                     blocks=4,
                     dilation_channels=128,
                     residual_channels=128,
                     skip_channels=512,
                     output_length=args.output_size,
                     dtype=dtype,
                     bias=True)
elif args.model == "sequnet":
    model = Sequnet(NUM_CLASSES, args.features, args.levels, NUM_CLASSES, kernel_size=5, target_output_size=args.output_size, depth=args.depth)
else:
    raise NotImplementedError("Could not find model " + str(args.model))

if args.cuda:
    model = DataParallel(model)
    print("move model to gpu")
    model.cuda()

print('model: ', model)
print('input size: ', model.input_size)
print('output size: ', model.output_size)
print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

writer = SummaryWriter(args.log_dir)

### DATASET
train_audio_data = AudioData(os.path.join(args.preprocessed_dataset_dir, "audio_train.hdf5"), sr=args.sr, channels=1)
if train_audio_data.is_empty():
    train_audio_data.add(glob.glob(os.path.join(args.dataset_dir, "train", "*.*")))
test_audio_data = AudioData(os.path.join(args.preprocessed_dataset_dir, "audio_test.hdf5"), sr=args.sr, channels=1)
if test_audio_data.is_empty():
    test_audio_data.add(glob.glob(os.path.join(args.dataset_dir, "test", "*.*")))

transform = lambda x : audio_to_onehot(x, model.output_size, NUM_CLASSES)
train_data = AudioDataset(train_audio_data, input_size=model.output_size, context_front=model.input_size-model.output_size+1, hop_size=40000, random_hops=True, audio_transform=transform)
print('Training dataset has ' + str(len(train_data)) + ' items')

# TRAINING
trainer = Trainer(model=model,
                  lr=args.lr,
                  eps=args.eps,
                  snapshot_folder=args.snapshot_dir,
                  logger=writer,
                  dtype=dtype,
                  ltype=ltype,
                  gradient_clipping=args.clip,
                  cuda=args.cuda,
                  num_workers=args.num_workers
                  )

print('start training...')
trainer.train(train_data, batch_size=args.batch_size, epochs=args.epochs)

# TESTING
print("start testing")
test_data= AudioDataset(test_audio_data, input_size=model.output_size, context_front=model.input_size-model.output_size+1, hop_size=model.output_size, random_hops=False, audio_transform=transform)
trainer.validate(test_data, batch_size=args.batch_size)
