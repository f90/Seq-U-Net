import torch
import torch.nn as nn
from sequnet_utils import Crop1d, Crop1dFrontBack

class ConvolutionBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, dropout, causal, activation=nn.LeakyReLU(), transpose=False):
        super(ConvolutionBlock, self).__init__()

        ops = list()
        if transpose:
            ops.append(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size, stride=2))

            if causal:
                crop_front = kernel_size - 1 - padding  # By default, crop at front and end to get only valid output, but crop less if padding is activated to get zero-padded outputs at start
                crop_back = kernel_size - 1
            else:
                assert (padding % 2 == 0)  # Non-causal: Crop less in front and back, equally
                crop_front = kernel_size - 1 - padding // 2
                crop_back = kernel_size - 1 - padding // 2

            ops.append(Crop1dFrontBack(crop_front, crop_back))

        else: # Normal convolution
            if padding > 0:
                if causal:
                    ops.append(torch.nn.ConstantPad1d((padding, 0), 0.0))
                else:
                    ops.append(torch.nn.ConstantPad1d((padding//2, padding//2), 0.0))

            ops.append(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride))

        if activation is not None:
            ops.append(activation)

        if dropout > 0.0:
            ops.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*ops)

    def forward(self, x):
        return self.block(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_shortcut, kernel_size, stride, padding, causal, dropout):
        super(UpsamplingBlock, self).__init__()

        # CONV 1 for UPSAMPLING
        self.conv1 = ConvolutionBlock(n_inputs, n_inputs, kernel_size, stride, padding, dropout, causal, transpose=True)

        # Crop operation for the shortcut connection that might have more samples!
        self.crop = Crop1d("front") if causal else Crop1d("both")

        # CONV 2 to combine high- with low-level information (from shortcut)
        self.conv2 = ConvolutionBlock(n_inputs + n_shortcut, n_outputs, kernel_size, 1, padding, dropout, causal)

    def forward(self, x, shortcut):
        upsampled = self.conv1(x)
        shortcut_crop = self.crop(shortcut, upsampled)
        combined = torch.cat([upsampled, shortcut_crop], 1)
        return self.conv2(combined)

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, causal, dropout):
        super(DownsamplingBlock, self).__init__()

        # CONV 1
        self.conv1 = ConvolutionBlock(n_inputs, n_outputs, kernel_size, 1, padding, dropout, causal)

        # CONV 2 with decimation
        self.conv2 = ConvolutionBlock(n_outputs, n_outputs, kernel_size, stride, padding, dropout, causal)

    def forward(self, x):
        shortcut = self.conv1(x)
        out = self.conv2(shortcut)
        return out, shortcut


class Sequnet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, kernel_size=3, causal=True, dropout=0.2, target_output_size=None):
        super(Sequnet, self).__init__()
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        self.num_levels = len(num_channels)
        self.kernel_size = kernel_size

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        # Handle padding
        self.set_output_size(target_output_size)

        for i in range(self.num_levels-1):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            self.downsampling_blocks.append(DownsamplingBlock(in_channels, num_channels[i], kernel_size, stride=2,
                                     padding=self.padding, dropout=dropout, causal=causal))

            self.upsampling_blocks.append(UpsamplingBlock(num_channels[i+1], num_channels[i], num_channels[i], kernel_size, stride=2, causal=causal,
                                                      padding=self.padding, dropout=dropout))

        self.bottleneck_conv = ConvolutionBlock(num_channels[-2], num_channels[-1], kernel_size, stride=1, causal=causal, padding=self.padding, dropout=dropout)
        self.output_conv = ConvolutionBlock(num_channels[0], num_outputs, 1, 1, 0, 0.0, False, None, False)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        if target_output_size is not None:
            self.padding = 0
            self.input_size, self.output_size = self.check_padding(target_output_size)
            print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")
        else:
            print("No target output size specified. Using zero-padded convolutions assuming input does NOT have further context! Input size = output size")
            self.padding = self.kernel_size - 1

    def check_padding(self, target_output_size):
        bottleneck_size = 2
        while True:
            out = self.check_padding_for_bottleneck(bottleneck_size, target_output_size)
            if out is not False:
                return out
            bottleneck_size += 1

    def check_padding_for_bottleneck(self, bottleneck_size, target_output_size):
        # Calculate output size with current bottleneck, check if its large enough, and if layer sizes on the way are correct
        curr_size = bottleneck_size
        for i in range(self.num_levels - 1):
            curr_size = curr_size * 2 - self.kernel_size + self.padding  # UpsampleConv
            if curr_size < 2: # We need at least two samples to interpolate
                return False
            curr_size = curr_size - self.kernel_size + 1 + self.padding # Conv
            if curr_size <  2 ** (i + 1): # All computational paths created from upsampling need to be covered
                return False

        output_size = curr_size
        if output_size < target_output_size:
            return False

        # Calculate input size with current bottleneck
        curr_size = bottleneck_size
        curr_size = curr_size + self.kernel_size - 1 - self.padding # Bottleneck-Conv
        for i in range(self.num_levels - 1):
            curr_size = curr_size*2 - 2 + self.kernel_size - self.padding # Strided conv
            if curr_size % 2 == 0: # Input to strided conv needs to have odd number of elements so we can keep the edge values in decimation!
                return False
            curr_size = curr_size + self.kernel_size - 1 - self.padding # Conv

        return curr_size, output_size

    def forward(self, x):
        curr_input_size = x.shape[-1]
        if self.target_output_size is None:
            # Input size = output size. Dynamically pad input so that we can provide outputs for all inputs
            self.input_size, self.output_size = self.check_padding(curr_input_size)
            # Pad input to required input size
            pad_op = torch.nn.ConstantPad1d((self.input_size - curr_input_size, 0), 0.0)
            x = pad_op(x)
        else:
            assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        # COMPUTE OUTPUT
        # DOWNSAMPLING BLOCKS
        shortcuts = list()
        out = x
        for block in self.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        out = self.bottleneck_conv(out)

        # UPSAMPLING BLOCKS
        for block, short in reversed(list(zip(self.upsampling_blocks, shortcuts))):
            out = block(out, short)

        # OUTPUT CONVOLUTION
        out = self.output_conv(out)

        # CROP OUTPUT, IF INPUT WAS PADDED EARLIER, TO MATCH SIZES
        if self.target_output_size is None:
            assert(out.shape[-1] == x.shape[-1]) # Output size = input size (including previous padding)
            # Crop output to required output size (since input was padded earlier)
            out = out[:, :,out.shape[-1] - curr_input_size:].contiguous()

        #print(out.shape)
        return out