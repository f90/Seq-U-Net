import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from copy_memory.utils import data_generator
from copy_memory.model import TCN
import time

import hyperopt
from hyperopt import fmin, tpe, hp

import numpy as np

parser = argparse.ArgumentParser(description='Sequence Modeling - Copying Memory Task')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 8)')
parser.add_argument('--iters', type=int, default=100,
                    help='number of iters per epoch (default: 100)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--blank_len', type=int, default=1000, metavar='N',
                    help='The size of the blank (i.e. T) (default: 1000)')
parser.add_argument('--seq_len', type=int, default=10,
                    help='initial history size (default: 10)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 50')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 10)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
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

def evaluate(data, labels):
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(range(0, data.shape[0], batch_size)):
            start_ind = batch
            end_ind = start_ind + batch_size

            x = data[start_ind:end_ind]
            y = labels[start_ind:end_ind]

            out = model(x)
            loss = criterion(out.view(-1, n_classes), y.view(-1))
            pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
            correct = pred.eq(y.data.view_as(pred)).cpu().sum()
            counter = out.view(-1, n_classes).size(0)
            acc = 100. * correct / counter

            losses.append(loss.item())
            accs.append(acc.item())

        loss = np.mean(np.array(losses))
        acc = np.mean(np.array(accs))
        print('\nTest set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(loss, acc))
        return loss, acc

def train(ep, lr, clip):
    global batch_size, seq_len, iters, epochs
    model.train()
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    for batch_idx, batch in enumerate(range(0, n_train, batch_size)):
        start_ind = batch
        end_ind = start_ind + batch_size

        x = train_x[start_ind:end_ind]
        y = train_y[start_ind:end_ind]
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, n_classes), y.view(-1))
        pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        counter += out.view(-1, n_classes).size(0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                ep, batch_idx, n_train // batch_size+1, lr, elapsed * 1000 / args.log_interval,
                avg_loss, 100. * correct / counter))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0

def optimize(init_lr, clip):
    print("TRAINING WITH LR " + str(init_lr) + ", CLIP " + str(clip))
    # Start training loop
    best_model_name = "model_" + args.experiment_name + ".pt"
    best_vloss = 1e8
    all_vloss = []
    lr = init_lr

    for epoch in range(1, epochs + 1):
        try:
            train(epoch, lr, clip)
        except OverflowError:
            return {'status': 'fail'}
        val_loss, _ = evaluate(val_x, val_y)

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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    batch_size = args.batch_size
    seq_len = args.seq_len  # The size to memorize
    epochs = args.epochs
    iters = args.iters
    T = args.blank_len
    n_steps = T + (2 * seq_len)
    n_classes = 10  # Digits 0 - 9
    n_train = 10000
    n_val = 1000
    n_test = 1000

    print(args)
    print("Preparing data...")
    train_x, train_y = data_generator(T, seq_len, n_train)
    val_x, val_y = data_generator(T, seq_len, n_val)
    test_x, test_y = data_generator(T, seq_len, n_test)

    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    model = TCN(args.model, 1, n_classes, channel_sizes, kernel_size, seq_len)
    print('Parameter count: ', str(sum(p.numel() for p in model.parameters())))

    if args.cuda:
        model.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

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
        train_loss, train_acc = evaluate(train_x, train_y)
        print('-' * 89)
        print('| End of training | train loss {:5.2f} | train acc {:8.2f}'.format(train_loss, train_acc))
        print('-' * 89)
        # Run on validation data
        val_loss, val_acc = evaluate(val_x, val_y)
        print('-' * 89)
        print('| End of training | val loss {:5.2f} | val acc {:8.2f}'.format(val_loss, val_acc))
        print('-' * 89)
        # Run on test data.
        test_loss, test_acc = evaluate(test_x, test_y)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test acc {:8.2f}'.format(test_loss, test_acc))
        print('=' * 89)
    else:
        print("ERROR: Training caused exploding loss due to instability!")
