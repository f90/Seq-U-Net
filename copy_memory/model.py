from torch import nn
import torch.nn.functional as F

from tcn import TemporalConvNet
from sequnet import Sequnet
from sequnet_res import Sequnet as SequnetRes


class TCN(nn.Module):
    def __init__(self, model, input_size, output_channels, num_channels, kernel_size, output_length):
        super(TCN, self).__init__()
        self.output_length = output_length
        self.model = model
        if model == "tcn":
            self.conv = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=0)
        elif model == "sequnet":
            self.conv = Sequnet(input_size, num_channels, num_channels[-1], kernel_size=kernel_size, dropout=0, target_output_size=output_length)
        elif model == "sequnet_res":
            self.conv = SequnetRes(input_size, num_channels[0], len(num_channels), num_channels[-1], kernel_size=kernel_size, target_output_size=output_length)
        else:
            raise NotImplementedError("Could not find this model " + model)

        self.linear = nn.Linear(num_channels[-1], output_channels)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.model == "tcn":
            y1 = self.conv(x)
        else:
            y1 = self.conv(F.pad(x, (self.conv.input_size - x.shape[2], 0), "constant", 0.0))
        y1 = y1[:,:, -self.output_length:]

        return self.linear(y1.transpose(1, 2))