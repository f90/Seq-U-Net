from torch import nn
from tcn import TemporalConvNet
import sequnet
import sequnet_res

class TCN(nn.Module):
    def __init__(self, args, num_characters, num_channels):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(num_characters, args.emsize)

        if args.model == "tcn":
            self.conv = TemporalConvNet(args.emsize, num_channels, kernel_size=args.ksize, dropout=args.dropout)
        elif args.model == "sequnet":
            self.conv = sequnet.Sequnet(args.emsize, num_channels, args.emsize, kernel_size=args.ksize, dropout=args.dropout, target_output_size=args.validseqlen)
            args.validseqlen = self.conv.output_size
            args.seq_len = self.conv.input_size
        elif args.model == "sequnet_res":
            self.conv = sequnet_res.Sequnet(args.emsize, num_channels[0], len(num_channels), args.emsize, kernel_size=args.ksize, target_output_size=args.validseqlen)
            args.validseqlen = self.conv.output_size
            args.seq_len = self.conv.input_size
        else:
            raise NotImplementedError("Could not find this model " + args.model)

        self.decoder = nn.Linear(args.emsize, num_characters)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(args.emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.conv(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()