import argparse
import time

import hyperopt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../")
from model import TCN
from utils import data_generator
import numpy as np
from hyperopt import fmin, tpe, hp

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate (default: 1e-2)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='Nott',
                    help='the dataset to run (default: Nott)')
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

criterion = nn.BCEWithLogitsLoss(reduction="sum")
input_size = 88
X_train, X_valid, X_test = data_generator(args.data)

torch.backends.cudnn.benchmark = True  # This makes dilated conv much faster for CuDNN 7.5

def evaluate(model, X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x.unsqueeze(0)).squeeze(0)
            loss = criterion(output, y)
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        print(name + " loss: {:.7f}".format(eval_loss))
        return eval_loss


def train(model, ep, lr, optimizer, clip):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    avg_time = 0
    if args.cuda:
        torch.cuda.reset_max_memory_allocated(device=torch.device("cuda"))

    for example_num, idx in enumerate(train_idx_list):
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        t = time.time()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = criterion(output, y)
        total_loss += loss.item()
        count += output.size(0)
        loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        t = time.time() - t
        avg_time += (1. / float(example_num+1)) *  (t - avg_time)

        if example_num > 0 and example_num % args.log_interval == 0:
            cur_loss = total_loss / count
            print("Epoch {:2d} | lr {:.7f} | loss {:.7f} | speed {:.7f}".format(ep, lr, cur_loss, avg_time))
            total_loss = 0.0
            count = 0
    if args.cuda:
        print("Max memory required by model: " + str(torch.cuda.max_memory_allocated(device=torch.device("cuda")) / (1024 * 1024)))

def optimize(lr, clip):
    print("Optimizing with " + str(lr) + "lr, " + str(args.epochs) + " epochs, " + str(clip) + " clip")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    print(args)

    n_channels = [args.nhid] * args.levels
    model = TCN(args.model, input_size, input_size, n_channels, args.ksize, dropout=args.dropout)
    print('Parameter count: ', str(sum(p.numel() for p in model.parameters())))

    if args.cuda:
        model.cuda()

    #summary(model, (193, 88))
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    best_vloss = 1e8
    vloss_list = []
    model_name = "model_" + str(args.data) + "_" + str(args.experiment_name) + ".pt"
    for ep in range(1, args.epochs + 1):
        train(model, ep, lr, optimizer, clip)
        vloss = evaluate(model, X_valid, name='Validation')
        if np.isnan(vloss) or vloss > 1000:
            return {'status' : 'fail'}
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-10:]):
            lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)
    return {'status' : 'ok', 'loss' : best_vloss, 'model_name' : model_name}

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
        best = hyperopt.space_eval(space, best)
        res = optimize(best[0], best[1])
        print("Reached validation loss of " + str(res["loss"]) + " with parameters:")
        print(best)

    # Test best model on train and test set
    print('-' * 89)
    model = torch.load(open(res["model_name"], "rb"))
    tloss = evaluate(model, X_train, name="Train")
    print('-' * 89)
    model = torch.load(open(res["model_name"], "rb"))
    tloss = evaluate(model, X_test)