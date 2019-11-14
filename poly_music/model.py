from torch import nn
from tcn import TemporalConvNet
import sequnet
import sequnet_res

class TCN(nn.Module):
    def __init__(self, model, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        if model == "tcn":
            self.conv = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
            self.linear = nn.Linear(num_channels[-1], output_size)
        elif model == "sequnet":
            self.conv = sequnet.Sequnet(input_size, num_channels, output_size, kernel_size, dropout=dropout)
        elif model == "sequnet_res":
            self.conv = sequnet_res.Sequnet(input_size, num_channels[-1], len(num_channels), output_size, kernel_size)
        else:
            raise NotImplementedError("Model not found: " + model)
        self.model_type = model

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if self.model_type == "tcn":
            output = self.linear(output)
        return output