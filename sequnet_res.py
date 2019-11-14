import torch
import torch.nn as nn
import numpy as np

from sequnet_utils import Crop1d, duplicate

class GroupNorm(nn.Module):
    def __init__(self, n_groups, n_channels, affine=True, eps=1e-5):
        super(GroupNorm, self).__init__()
        assert(n_channels % n_groups == 0)
        if affine:
            self.gamma = nn.Parameter(torch.ones((n_channels)))
            self.beta = nn.Parameter(torch.zeros((n_channels)))

        self.affine = affine
        self.mul_apply = MulConstant().apply
        self.eps = eps
        self.n_groups = n_groups
        self.channels_per_group = n_channels // n_groups

    def forward(self, x):
        shape = list(x.shape)

        x = x.view(shape[:1] + [self.n_groups, self.channels_per_group] + shape[2:])
        dims = [i for i in range(2, len(x.shape))]
        means = torch.mean(x, dim=dims, keepdim=True)
        factor = 1.0 / torch.sqrt(torch.var(x, dim=dims, unbiased=False, keepdim=True) + self.eps)

        x = x - means
        self.mul_apply(x, factor)
        x = x.view(shape)

        if self.affine:
            self.mul_apply(x, self.gamma.view(1, -1, 1))
            return x + self.beta.view(1, -1, 1)

        return x

class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, factors):
        # ctx is a context object that can be used to stash information
        # for backward computation
        #tensor.add_(-means)
        tensor.mul_(factors)
        ctx.mul = factors

        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.

        return grad_output * ctx.mul

class ResidualLayer(nn.Module):
    def __init__(self, cycles, n_inputs, n_outputs, kernel_size, stride, transpose=False):
        super(ResidualLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.cycles = cycles

        conv = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.filter = conv(n_inputs, n_outputs, kernel_size, stride)
        self.gate = conv(n_inputs, n_outputs, kernel_size, stride)
        self.identity_crop = Crop1d("front")

    def forward(self, x, mode, clock, conditional=None):
        if x is None:
            assert(mode == "step") # No new input - we have to be in step mode
            # If we are not an upsampling layer, or if we are but its not our cycle, don't need to compute anything
            if (not self.transpose) or (clock % self.cycles != 0):
                return None
            else: # Otherwise return the previously computed output value if the clock says so
                return self.next_output

        # Update inputs
        if mode == "save": # Save inputs, if desired
            self.inputs = x[:,:,-self.kernel_size:]
            if conditional is not None:
                self.conditional = conditional[:,:,-self.kernel_size:]
            else:
                self.conditional = None
        elif mode == "step": # Update inputs
            self.inputs = torch.cat([self.inputs[:,:,1:], x], dim=2)
            if conditional is not None:
                self.conditional = torch.cat([self.conditional[:, :, 1:], conditional], dim=2)

            x = self.inputs
            conditional = self.conditional

            # If we have a normal or strided conv and it is not our cycle, we don't need to return anything
            if clock % self.cycles != 0 and not self.transpose:
                return None

        # BEGINNING OF RESIDUAL BLOCK COMPUTATION
        # If we have conditional input, concatenate it so the convolution operates on the input and conditional
        if conditional is None:
            res_input = x
        else:
            res_input = torch.cat([x, conditional], dim=1)

        # Apply the residual convolutions
        filter = torch.tanh(self.filter(res_input))
        gate = torch.sigmoid(self.gate(res_input))
        residual = filter * gate

        # Crop if transposed conv was applied
        if self.transpose:
            # By default, crop at front and end to get only valid output, but crop less if padding is activated to get zero-padded outputs at start
            if mode != "step":
                crop_front = self.kernel_size - 1 - self.pad_front
            else:
                crop_front = self.kernel_size - 1

            if mode == "step" or mode == "save":
                crop_back = self.kernel_size - 2 # Crop one less at the end since this will be our "future" value we will output based on padding an extra zero into input
            else:
                crop_back = self.kernel_size - 1

            if crop_back > 0:
                residual = residual[:,:,crop_front:-crop_back].contiguous()
            else:
                residual = residual[:, :, crop_front:].contiguous()

        # PREPARE IDENTITY CONNECTION
        if self.transpose:
            # UPSAMPLING CASE
            if mode == "step" or mode == "save":
                identity = torch.cat([duplicate(x), x[:,:,-1:]], dim=2) # Need to duplicate last value as well!
            else:
                identity = duplicate(x)
        else:
            if self.stride > 1:
                # DOWNSAMPLING CASE
                identity = x[:,:,(x.shape[2]-1)%2::self.stride] # Make sure to always take the very last output here
            else:
                identity = x

        # Crop identity connection to fit residual, in case residual got smaller due to more convolutions
        identity = self.identity_crop(identity, residual)

        # IDENTITY AND RESIDUAL READY
        out = identity + residual
        if (mode == "step" or mode == "save") and self.transpose:
            # Need to save the "future" output that was created with zero-padding
            self.next_output = out[:,:,-1:]
            if mode == "step":
                out = out[:,:,-2:-1] # Keep only second-last output for step mode
            else:
                out = out[:,:,:-1] # Get all outputs except the extra "future" one in save mode
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose and self.stride > 1:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        curr_size = curr_size + self.kernel_size - 1

        # Transposed
        if self.transpose:
            self.pad_front = (curr_size - 1) % self.stride # Only need to pad if after removing first element, we cannot divide by stride evenly
            curr_size = int(np.ceil(curr_size / self.stride)) # If zero-padding led to e.g. 5 outputs, there must be ceil(5/2) = 3 elements involved. For 4 it would be 2.
        assert(curr_size > 0)
        return curr_size

class UpsamplingBlock(nn.Module):
    def __init__(self, cycles, n_channels, n_shortcut, kernel_size, stride, depth):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        self.upconv = ResidualLayer(cycles, n_channels, n_channels, kernel_size, stride, transpose=True)

        # Crop operation for the shortcut connection that might have more samples!
        self.crop = Crop1d("front")

        # CONVS to combine high- with low-level information (from shortcut)
        self.convs = nn.ModuleList([ResidualLayer(cycles, n_channels + n_shortcut, n_channels, kernel_size, 1) for _ in range(depth)])

    def forward(self, x, mode, clock, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x, mode, clock)

        # Prepare shortcut connection
        combined = self.crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.convs:
            combined = conv(combined, mode, clock, self.crop(upsampled, combined))
        return combined

    def get_input_size(self, output_size):
        curr_size = output_size
        # Combine convolutions
        for conv in reversed(self.convs):
            curr_size = conv.get_input_size(curr_size)
        # Upsampling conv
        return self.upconv.get_input_size(curr_size)

class DownsamplingBlock(nn.Module):
    def __init__(self, cycles, n_channels, kernel_size, stride, depth):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride
        self.crop = Crop1d("front")

        # CONV 1
        self.convs = nn.ModuleList(
            [ResidualLayer(cycles, n_channels, n_channels, kernel_size, 1) for _ in range(depth)])

        # CONV 2 with decimation
        self.downconv = ResidualLayer(cycles * 2, n_channels, n_channels, kernel_size, stride)

    def forward(self, x, mode, clock, prev_features=None):
        if prev_features is None:
            shortcut = x
            for conv in self.convs:
                shortcut = conv(shortcut, mode, clock)
        else:
            shortcut = self.crop(prev_features, x) # Features have become smaller since we last saved identity features
            for conv in self.convs:
                shortcut = conv(shortcut, mode, clock, self.crop(x, shortcut))
        out = self.downconv(shortcut, mode, clock)
        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)
        for conv in reversed(self.convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Sequnet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_levels, num_outputs, kernel_size=3, target_output_size=None, depth=1):
        super(Sequnet, self).__init__()
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        self.bottlenecks = nn.ModuleList()

        self.num_levels = num_levels
        self.kernel_size = kernel_size
        # For fast causal inference only
        self.clock = 1
        self.cycles = 2 ** (self.num_levels - 1)

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        self.input_conv = nn.Conv1d(num_inputs, num_channels, kernel_size, stride=1)
        self.crop = Crop1d("front")

        for i in range(self.num_levels - 1):
            self.downsampling_blocks.append(
                DownsamplingBlock(2**i, num_channels, kernel_size, 2, depth=depth))
            self.upsampling_blocks.append(
                UpsamplingBlock(2**i, num_channels, num_channels, kernel_size, 2, depth=depth))

        self.bottlenecks = nn.ModuleList(
            [ResidualLayer(2 ** (self.num_levels - 1), num_channels, num_channels, kernel_size, 1) for _ in range(depth)])

        output_ops = [nn.ReLU(), nn.Conv1d(num_channels, num_outputs, 1, 1)]
        self.output_conv = nn.Sequential(*output_ops)

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        if target_output_size is not None:
            self.input_size, self.output_size = self.check_padding(target_output_size)
            print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")
        else:
            print("No target output size specified. Using zero-padded convolutions assuming input does NOT have further context! Input size = output size")

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        output_size = (target_output_size // self.cycles) * self.cycles
        if target_output_size % self.cycles > 0:
            output_size += self.cycles

        while True:
            out = self.check_padding_for_bottleneck(output_size)
            if out is not False:
                return out
            output_size += self.cycles

    def check_padding_for_bottleneck(self, target_output_size):
        try:
            curr_size = target_output_size
            # Calculate output size with current bottleneck, check if its large enough, and if layer sizes on the way are correct
            for block in self.upsampling_blocks:
                curr_size = block.get_input_size(curr_size)
            # Bottleneck-Conv
            for block in reversed(self.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for block in reversed(self.downsampling_blocks):
                curr_size = block.get_input_size(curr_size)

            curr_size = curr_size + self.kernel_size - 1 # Input conv

            return curr_size, target_output_size
        except Exception as e:
            return False

    def forward(self, x, mode="normal"):
        if mode == "normal" or mode == "save":
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
        if mode == "save":
            self.inputs = x[:,:,-self.kernel_size:]
        elif mode == "step":
            self.inputs = torch.cat([self.inputs[:,:,1:], x], dim=2)
            x = self.inputs

        out = self.input_conv(x)

        # DOWNSAMPLING BLOCKS
        shortcuts = list()
        for block in self.downsampling_blocks:
            out, short = block(out, mode, self.clock)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in self.bottlenecks:
            out = conv(out, mode, self.clock)

        # UPSAMPLING BLOCKS
        for block, short in reversed(list(zip(self.upsampling_blocks, shortcuts))):
            out = block(out, mode, self.clock, short)

        # OUTPUT CONVOLUTION (has no memory)
        out = self.output_conv(out)

        if mode == "normal" or mode == "save":
            # Output size might not match targets if we are in "input=output size mode"
            if self.target_output_size is None:
                out = out[:, :, -curr_input_size:].contiguous() # Take end of output
        else:
            self.clock = (self.clock + 1) % self.cycles

        return out