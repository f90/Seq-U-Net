import os

import torch
import torch.utils.data
import time
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

import utils
from utils import reshape_output

class Trainer:
    def __init__(self,
                 model,
                 logger,
                 lr,
                 eps,
                 cuda,
                 gradient_clipping=None,
                 snapshot_folder=None,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor,
                 num_workers=1):
        self.model = model
        self.lr = lr
        self.clip = gradient_clipping
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr, eps=eps)
        self.logger = logger
        self.snapshot_folder = snapshot_folder
        self.dtype = dtype
        self.ltype = ltype
        self.cuda = cuda
        self.num_workers = num_workers

    def train(self,
              dataset,
              batch_size=16,
              epochs=10,
              load_snapshot=None):

        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True,num_workers=self.num_workers)

        step = 0
        # LOAD MODEL CHECKPOINT IF DESIRED
        if load_snapshot is not None:
            print("Loading model from checkpoint " + str(load_snapshot))
            step = utils.load_model(self.model, self.optimizer, os.path.join(self.snapshot_folder, load_snapshot))

        self.model.train()

        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            avg_time = 0.
            with tqdm(total=len(dataset) // batch_size) as pbar:
                for example_num, (x, target) in enumerate(dataloader):
                    if self.cuda:
                        torch.cuda.reset_max_memory_allocated(device=torch.device("cuda"))
                        x = x.cuda()
                        target = target.cuda()
                    target = target.view(-1)

                    t = time.time()

                    output = self.model(x)
                    output = reshape_output(output)
                    loss = F.cross_entropy(output.squeeze(), target)
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.clip is not None and self.clip > 0.0:
                        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    step += 1

                    t = time.time() - t
                    avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                    if step == 100:
                        print("Average time per training step: " + str(avg_time) + " seconds)")
                        if self.cuda:
                            print("Max memory required by model: " + str(
                                torch.cuda.max_memory_allocated(device=torch.device("cuda")) / (1024 * 1024)))

                    accuracy = torch.argmax(output, 1).eq(target).sum().float() / len(target)
                    self.logger.add_scalar("train_accuracy", accuracy.item(), step)
                    self.logger.add_scalar("train_loss", loss.item(), step)

                    pbar.update(1)

            # SNAPSHOT
            if self.snapshot_folder is not None:
                print("Saving model...")
                utils.save_model(self.model, self.optimizer, step, os.path.join(self.snapshot_folder, "checkpoint_" + str(step)))


    def validate(self, dataset, batch_size=16, load_snapshot=None, temperature=1.0):
        # PREPARE DATA
        dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=4)

        # LOAD SNAPSHOT
        if load_snapshot is not None:
            print("Loading model from checkpoint " + str(load_snapshot))
            utils.load_model(self.model, None, os.path.join(self.snapshot_folder, load_snapshot))

        # VALIDATE
        self.model.eval()
        total_loss = []
        accuracies = []
        samples = []
        with tqdm(total=len(dataset)//batch_size) as pbar, torch.no_grad():
            for (x, target) in dataloader:
                if self.cuda:
                    x = x.cuda()
                    target = target.cuda()
                target = target.view(-1)

                output = self.model(x)
                output = reshape_output(output)
                loss = F.cross_entropy(output.squeeze() / temperature, target)
                total_loss.append(loss.item())

                predictions = torch.argmax(output, 1).view(-1)
                correct_pred = torch.eq(target, predictions)
                accuracy = torch.sum(correct_pred).item() / float(len(target))
                accuracies.append(accuracy)

                samples.append(len(target))
                pbar.update(1)
                pbar.set_description("Bits: {:0.2f}".format(loss.item() * np.log2(np.e)))

        samples = np.array(samples) / np.sum(np.array(samples))
        avg_loss = np.sum(np.multiply(np.array(total_loss), samples))
        avg_accuracy = np.sum(np.multiply(np.array(accuracies), samples))
        avg_bits = avg_loss * np.log2(np.e)
        self.model.train()

        print("VALIDATION FINISHED: NLL: " + str(avg_loss) + ", ACC: " + str(avg_accuracy) + ", BITS: " + str(avg_bits))
        self.logger.add_scalar("test_accuracy", avg_accuracy, 0)
        self.logger.add_scalar("test_loss", avg_loss, 0)
        self.logger.add_scalar("test_bits_per_sample", avg_bits, 0)
        return avg_loss, avg_accuracy, avg_bits