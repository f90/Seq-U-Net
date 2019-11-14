from torch import nn
import sys
sys.path.append("..")
from tcn import TemporalConvNet
import sequnet
import sequnet_res

class TCN(nn.Module):

    def __init__(self, args, num_words, num_channels):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(num_words, args.emsize)

        if args.model == "tcn":
            self.conv = TemporalConvNet(args.emsize, num_channels, kernel_size=args.ksize, dropout=args.dropout)
        elif args.model == "sequnet" or args.model == "sequnet_res":
            if args.model == "sequnet":
                self.conv = sequnet.Sequnet(args.emsize, num_channels, args.emsize, kernel_size=args.ksize,
                                             dropout=args.dropout, target_output_size=args.validseqlen)
            else:
                self.conv = sequnet_res.Sequnet(args.emsize, num_channels[0], len(num_channels), args.emsize,
                                                kernel_size=args.ksize, target_output_size=args.validseqlen)
            args.validseqlen = self.conv.output_size
            args.seq_len = self.conv.input_size
            print("Using Seq-U-Net with " + str(args.validseqlen) + " outputs and " + str(args.seq_len) + " inputs")
        else:
            raise NotImplementedError("Could not find this model " + args.model)

        self.decoder = nn.Linear(num_channels[-1], num_words)
        if args.tied:
            if num_channels[-1] != args.emsize:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(args.emb_dropout)
        self.emb_dropout = args.emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.conv(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()