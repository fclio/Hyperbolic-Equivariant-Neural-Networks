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
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()


class SplitGConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1,
                 pad=0, wscale = 0.1, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
        super(SplitGConv2D, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = ksize

        kernel_size = _pair(ksize)
        stride = _pair(stride)
        padding = _pair(pad)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size
        self.wscale = wscale

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(wscale)

        self.inds = self.make_transformation_indices()

    # def reset_parameters(self):
    #     n = self.in_channels
    #     for k in self.kernel_size:
    #         n *= k
    #     stdv = 1. / math.sqrt(n)
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)
    def reset_parameters(self, wscale=1.0):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = wscale / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds)
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)

        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y


class P4ConvZ2(SplitGConv2D):
    """
    for 90 degree rotation, so 4 group
    from 1 input to 4 diff output feature map
    """

    def __init__(self, *args, **kwargs):
        super(P4ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, *args, **kwargs)


class P4ConvP4(SplitGConv2D):
    """
    for 90 degree rotation, so 4 group
    from 4 input to 4 diff output feature map
    """
    def __init__(self, *args, **kwargs):
        super(P4ConvP4, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, *args, **kwargs)


class P4MConvZ2(SplitGConv2D):
    """
    for 90 degree rotation and reflection, so 4 rotation group + its own reflection= 8 group
    from 1 input to 8 diff output feature map
    """
    def __init__(self, *args, **kwargs):
        super(P4MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, *args, **kwargs)


class P4MConvP4M(SplitGConv2D):
    """
    for 90 degree rotation and reflection, so 4 rotation group + its own reflection= 8 group
    from 8 input to 8 diff output feature map
    """
    def __init__(self, *args, **kwargs):
        super(P4MConvP4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)