import argparse
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import TCN
from utils import data_generator, batchify, get_batch

import hyperopt
from hyperopt import fmin, tpe, hp
from hyperopt.pyll.base import scope

parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.5)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.4,
                    help='gradient clip, -1 means no clip (default: 0.4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--emsize', type=int, default=600,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=45,
                    help='valid sequence length (default: 45)')
parser.add_argument('--seq_len', type=int, default=117,
                    help='total sequence length, including effective history (default: 117)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
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
corpus = data_generator(args)
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, args.batch_size, args)
test_data = batchify(corpus.test, args.batch_size, args)

n_words = len(corpus.dictionary)

# Check sequence length setting
eff_history = args.seq_len - args.validseqlen
if eff_history < 0:
    raise ValueError("Valid sequence length must be smaller than sequence length!")

# May use adaptive softmax to speed up training
criterion = nn.CrossEntropyLoss()

def compute_loss(predictions, targets):
    eff_history = args.seq_len - args.validseqlen
    final_target = targets[:, eff_history:].contiguous()
    if final_target.shape[-1] < predictions.shape[1]:
        predictions = predictions[:, eff_history:].contiguous()
    loss = criterion(predictions.view(-1, n_words), final_target.view(-1))
    return loss

def evaluate(model, data_source):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    with torch.no_grad():
        for i in range(0, data_source.size(1) - 1, args.validseqlen):
            if i + args.seq_len >= data_source.size(1) - 1:
                continue
            data, targets = get_batch(data_source, i, args)
            output = model(data)

            # Discard the effective history, just like in training
            loss = compute_loss(output, targets)

            # Note that we don't add TAR loss here
            total_loss += (data.size(1) - eff_history) * loss.item()
            processed_data_size += data.size(1) - eff_history
        return total_loss / processed_data_size


def train(model, optimizer, lr, epoch, clip):
    print("Training... ")
    model.train() # Turn on training mode which enables dropout.
    total_loss = 0
    start_pos = [i for i in range(0, train_data.size(1) - 1, args.validseqlen) if i + args.seq_len < train_data.size(1)]
    random.shuffle(start_pos)
    avg_time = 0.
    if args.cuda:
        torch.cuda.reset_max_memory_allocated(device=torch.device("cuda"))

    for batch_idx, i in enumerate(start_pos):
        data, targets = get_batch(train_data, i, args)
        optimizer.zero_grad()

        t = time.time()

        output = model(data)
        loss = compute_loss(output, targets)
        loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

        t = time.time() - t
        avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | time/it {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, train_data.size(1) // args.validseqlen, lr,
                avg_time, cur_loss, math.exp(cur_loss)))
            total_loss = 0
    if args.cuda:
        print("Max memory required by model: " + str(
            torch.cuda.max_memory_allocated(device=torch.device("cuda")) / (1024 * 1024)))

def optimize(lr, clip):
    print("Optimizing with " + str(lr) + "lr, " + str(args.epochs) + " epochs, " + str(clip) + " clip")

    num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
    model = TCN(args, n_words, num_chans)

    if args.cuda:
        model.cuda()

    print("Parameters: " + str(sum(p.numel() for p in model.parameters())))
    torch.backends.cudnn.benchmark = True  # This makes dilated conv much faster for CuDNN 7.5

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    # Start training loop
    best_model_name = "model_" + args.experiment_name + ".pt"
    best_vloss = 1e8

    all_vloss = []
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        try:
            train(model, optimizer, lr, epoch, clip)
        except OverflowError:
            return {'status': 'fail'}

        print("Validating...")
        val_loss = evaluate(model, val_data)
        if np.isnan(val_loss) or val_loss > 100:
            return {'status' : 'fail'}

        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))

        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_vloss:
            with open(best_model_name, 'wb') as f:
                print('Save model!\n')
                torch.save(model, f)
            best_vloss = val_loss

        # Anneal the learning rate if the validation loss plateaus
        if epoch > 10 and val_loss >= max(all_vloss[-5:]):
            lr = lr / 2.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        all_vloss.append(val_loss)

    return {"status" : "ok", "loss" : best_vloss, "model_name" : best_model_name}

if __name__ == "__main__":
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

    # Load the best saved model.
    if res["status"] != "fail":
        with open(res["model_name"], 'rb') as f:
            model = torch.load(f)

        # Run on train data
        train_loss = evaluate(model, train_data)
        print('-' * 89)
        print('| End of training | train loss {:5.2f} | train ppl {:8.2f}'.format(
            train_loss, math.exp(train_loss)))
        print('-' * 89)
        # Run on validation data
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| End of training | validation loss {:5.2f} | validation ppl {:8.2f}'.format(
            val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Run on test data.
        test_loss = evaluate(model, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
    else:
        print("ERROR: Training caused exploding loss due to instability!")