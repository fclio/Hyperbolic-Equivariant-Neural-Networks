import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair
from groupy.gconv.make_gconv_indices import *

make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices}


def trans_filter(w, inds):
    
    #  self.weight: torch.Size([64, 1, 1, 3, 3])
    #  (out_channels, in_channels, input_stabilizer_size, kernel_size, kernel_size)
    #  self.inds: (8, 1, 3, 3, 3)
    #  (num_group_transformation, input_stabilizer_size, kernel_size, kernel_size, 3)
       
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    # indeces: (8 * 1 * 3 * 3, 3) = (72, 3)
    # Each row contains a (T, U, V) triplet that maps a pixel from the original kernel to a transformed version.

    #  step 2: this remaps the filter weights based on the transformation indices.
    # !!! here is a special index finding will become w(:,:, len(n-dimension_list matching follow select base on index))
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    # Extracts transformed filter pixels using (T, U, V) from w
    # w_indexed.shape = (64, 1, 72) (since 72 = 8 * 1 * 3 * 3)
    # (out_channels, in_channels, num_group_transformation * input_stabilizer_size * kernel_size * kernel_size)

    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    # torch.Size([64, 1, 8, 1, 3, 3])
    # (out_channels, in_channels, num_group_transformation, input_stabilizer_size, kernel_size, kernel_size)


    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    #  torch.Size([64, 8, 1, 1, 3, 3])
    # (out_channels, num_group_transformation, in_channels, input_stabilizer_size, kernel_size, kernel_size)
    

    # Ensures that the tensor is stored contiguously in memory
    # This improves efficiency for PyTorch computations
    return w_transformed.contiguous()


class SplitGConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
        super(SplitGConv2D, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        # here all filter is learnable
        self.weight = Parameter(torch.Tensor(
            self.out_channels, in_channels, self.input_stabilizer_size, *kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

        # contains a list of index mappings for transformations like rotation and reflection. 
        # Instead of manually shifting pixels, we use indices to reorder the filter weights efficiently.
        # !!! for each of the 8 transformations, it gives a mapping (T, U, V) for every pixel in the 3x3 kernel.
        self.inds = self.make_transformation_indices()
        # (8, 1, 3, 3, 3)
        # (Number of output transformations (D4 group: 4 rotations + 4 reflections)
        # Number of input transformations (Z2 space: only 1 channel, no structured symmetries)
        # Kernel size(3x3), 
        # 3D indices (T, U, V) for transformation mappin
        ## Transformation Index: 
        # For P4 (Rotational Group):
        # T = 0 → 0° Rotation (identity)
        # T = 1 → 90° Rotation
        # T = 2 → 180° Rotation
        # T = 3 → 270° Rotation
        # For P4M (Rotations + Reflections):
        # T = 4 → Horizontal Reflection
        # T = 5 → Vertical Reflection
        # T = 6 → Diagonal Reflection (↘)
        # T = 7 → Diagonal Reflection (↙)
        # U (Row Index in Kernel Space)
        # V (Column Index in Kernel Space)


    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        # to understand later!
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        # step 1: Transforms filters using the precomputed transformation indices.
        # change the self.weight 
        # print("before", self.weight)
        # print(self.weight)

        #  self.weight: torch.Size([64, 1, 1, 3, 3])
        #  (out_channels, in_channels, input_stabilizer_size, kernel_size, kernel_size)
        #  self.inds: (8, 1, 3, 3, 3)
        #  (num_group_transformation, input_stabilizer_size, kernel_size, kernel_size, 3)
        tw = trans_filter(self.weight, self.inds)
        # torch.Size([64, 8, 1, 1, 3, 3])
        # (out_channels, num_group_transformation, in_channels, input_stabilizer_size, kernel_size, kernel_size)
        
                
        # Reshapes transformed weights for use in conv2d.
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)
        # print("tw information")
        # print(isinstance(tw, nn.Parameter))  # This will print False
        # print(tw.requires_grad)
        # Reshapes input to match transformed kernel dimensions.
        input_shape = input.size()

        # before group equivairn input: [batch size, in_channels, kernel size, kernel size]
        # after group equivairn input: [batch size, in_channels, input_stabilizer_size, kernel size, kernel size]
        # !!!! this is important since before 3* 1= 3 it just keep the same shape
        # it is mainly use later to concatneate all num_transformation into one list, as including each transformation as additional input channel so instead of rgb, but rgb for each transformation
        
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])
        # after group equivairn input: [batch size, in_channels * input_stabilizer_size, kernel size, kernel size]
 
        # print("learnable weight", tw.requires_grad)  # Should be True

        # weight: ([513, 1, 3, 3]) (out_channel, in_channel, ksize, ksize)
        # weight determine the out_channel for output
        # input: ([128, 1, 32, 32]) (batch_size，in_channel, img_size, img_size)
        # input determine the batch
        # Performs convolution using the transformed weights and reshaped input
        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        
        # if doing lorenz + equivariant, then i must directly do nn.linear, 
       
        # with dilation:
        #  h_out = math.floor(
        #     (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        # w_out = math.floor(
        #     (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        # without:
        #  h_out = math.floor(
        #     (h + 2 * self.padding[0] -  self.kernel_size[0]) / self.stride[0] + 1)
        # w_out = math.floor(
        #     (w + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)

        batch_size, _, ny_out, nx_out = y.size()
        # print("y_size", y.size())
        # output: ([128, 512, 16, 16]) (batch, out_channel, ny, nx)
       
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
      
        # ([128, 64, 8, 16, 16]) (batch_size, out_channels * output_stab, )
        # print(y.view(batch_size,  self.out_channels *self.output_stabilizer_size* ny_out* nx_out))
        # print(self.out_channels *self.output_stabilizer_size* ny_out* nx_out)

        
        # here why using this shape? why having out_channel, then output_size? for equivairant?
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y


class P4ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, *args, **kwargs)


class P4ConvP4(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4ConvP4, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, *args, **kwargs)


class P4MConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, *args, **kwargs)


class P4MConvP4M(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4MConvP4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)