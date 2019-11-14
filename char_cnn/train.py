import argparse
import random

import hyperopt
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.append("../")
from utils import *
from model import TCN
import time
import math
from hyperopt import fmin, tpe, hp
import numpy as np

import warnings
warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode


parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to the embedded layer (0 = no dropout) (default: 0.1)')
parser.add_argument('--clip', type=float, default=0.4,
                    help='gradient clip, -1 means no clip (default: 0.15)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--emsize', type=int, default=100,
                    help='dimension of character embeddings (default: 100)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--nhid', type=int, default=450,
                    help='number of hidden units per layer (default: 450)')
parser.add_argument('--validseqlen', type=int, default=320,
                    help='valid sequence length (default: 320)')
parser.add_argument('--seq_len', type=int, default=400,
                    help='total sequence length, including effective history (default: 400)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to use (default: ptb)')
parser.add_argument('--model', type=str, default='tcn',
                    help='Model to use (tcn/sequnet)')
parser.add_argument('--hyper_iter', type=int, default=20,
                    help='Hyper-parameter optimisation runs (default: 20)')
parser.add_argument('--mode', type=str, default='hyper',
                    help='Training mode - hyper: Perform hyper-parameter optimisation, return best configuration. '
                         'train: Train single model with given parameters')
parser.add_argument('--experiment_name', type=str, default=str(np.random.randint(0, 100000)),
                    help="Optional name of experiment, used for saving model checkpoint")

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator(args)

n_characters = len(corpus.dict)
train_data = batchify(char_tensor(corpus, file), args.batch_size, args)
val_data = batchify(char_tensor(corpus, valfile), 1, args)
test_data = batchify(char_tensor(corpus, testfile), 1, args)
print("Corpus size: ", n_characters)

criterion = nn.CrossEntropyLoss()

def compute_loss(predictions, targets):
    eff_history = args.seq_len - args.validseqlen
    final_target = targets[:, eff_history:].contiguous()
    if final_target.shape[-1] < predictions.shape[1]:
        predictions = predictions[:, eff_history:].contiguous()
    loss = criterion(predictions.view(-1, n_characters), final_target.view(-1))
    return loss

def evaluate(model, source):
    model.eval()
    total_loss = 0
    count = 0
    source_len = source.size(1)
    with torch.no_grad():
        for batch, i in enumerate(range(0, source_len - 1, args.validseqlen)):
            if i + args.seq_len >= source_len:
                continue
            inp, target = get_batch(source, i, args)
            output = model(inp)
            loss = compute_loss(output, target)
            total_loss += loss.data * args.validseqlen
            count += args.validseqlen

        val_loss = total_loss.item() / count * 1.0
        return val_loss


def train(model, optimizer, clip, lr, epoch):
    model.train()
    total_loss = 0
    losses = []
    source = train_data
    source_len = source.size(1)
    avg_time = 0.
    if args.cuda:
        torch.cuda.reset_max_memory_allocated(device=torch.device("cuda"))

    # Shuffle start indices so the training data is shuffled
    start_pos = [i for i in range(0, source_len - 1, args.validseqlen) if i + args.seq_len < source_len]
    random.shuffle(start_pos)
    for batch_idx, i in enumerate(start_pos):
        inp, target = get_batch(source, i, args)
        optimizer.zero_grad()

        t = time.time()

        output = model(inp)
        loss = compute_loss(output, target)
        loss.backward()

        if np.isnan(loss.item()) or loss.item() > 100:
            raise OverflowError("Training failed due to loss explosion")

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

        t = time.time() - t
        avg_time += (1. / float(batch_idx+1)) *  (t - avg_time)

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            losses.append(cur_loss)
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | time {:5.5f} | '
                  'loss {:5.3f} | bpc {:5.3f}'.format(
                epoch, batch_idx, int((source_len-0.5) / args.validseqlen), lr,
                avg_time, cur_loss, cur_loss / math.log(2)))
            total_loss = 0

    if args.cuda:
        print("Max memory required by model: " + str(
            torch.cuda.max_memory_allocated(device=torch.device("cuda")) / (1024 * 1024)))
    return sum(losses) * 1.0 / len(losses)

def optimize(lr, clip):
    print("Optimizing with " + str(lr) + "lr, " + str(args.epochs) + " epochs, " + str(clip) + " clip")
    num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
    model = TCN(args, n_characters, num_chans)

    if args.cuda:
        model.cuda()

    print("Parameters: " + str(sum(p.numel() for p in model.parameters())))
    torch.backends.cudnn.benchmark = True  # This makes dilated conv much faster for CuDNN 7.5
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    # Start training loop
    all_losses = []
    best_vloss = 1e7
    for epoch in range(1, args.epochs + 1):
        try:
            train(model, optimizer, clip, lr, epoch)
        except OverflowError:
            return {'status': 'fail'}

        vloss = evaluate(model, val_data)
        if np.isnan(vloss) or vloss > 1000:
            return {'status' : 'fail'}
        print('-' * 89)
        print('| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}'.format(
            epoch, vloss, vloss / math.log(2)))

        if epoch > 10 and vloss > max(all_losses[-5:]):
            lr = lr / 2.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        all_losses.append(vloss)

        if vloss < best_vloss:
            print("Saving...")
            with open("model_" + args.experiment_name + ".pt", "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
    return {"status" : "ok", "loss" : best_vloss, "model_name" : "model_" + args.experiment_name + ".pt"}


def main():
    if args.mode == "train": # Just train with given parameters
        res = optimize(args.lr, args.clip)
    else:
        assert(args.mode == "hyper")
        space = [hp.loguniform('lr', -12, -2),
                 hp.choice('clip_decision', [-1, hp.uniform('clip', 0.01, 1.0)])]

        best = fmin(fn=lambda args : optimize(args[0], args[1]),
                    space=space,
                    algo=tpe.suggest,
                    max_evals=args.hyper_iter)

        print("Train model with best settings...")
        best = hyperopt.space_eval(space, best)
        res = optimize(best[0], best[1])
        print("Reached validation loss of " + str(res["loss"]) + " with parameters:")
        print(best)

    # Load best model
    model = torch.load(open(res["model_name"], "rb"))
    # Run on train data
    train_loss = evaluate(model, train_data)
    print('=' * 89)
    print('| End of training | train loss {:5.3f} | train bpc {:8.3f}'.format(
        train_loss, train_loss / math.log(2)))
    print('=' * 89)
    # Run on validation data
    val_loss = evaluate(model, val_data)
    print('=' * 89)
    print('| End of training | val loss {:5.3f} | val bpc {:8.3f}'.format(
        val_loss, val_loss / math.log(2)))
    print('=' * 89)
    # Run on test data.
    test_loss = evaluate(model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
        test_loss, test_loss / math.log(2)))
    print('=' * 89)

# train_by_random_chunk()
if __name__ == "__main__":
    main()
