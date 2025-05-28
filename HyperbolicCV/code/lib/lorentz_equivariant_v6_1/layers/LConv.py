import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz_equivariant_v6_1.layers.LFC import GroupLorentzFullyLinear
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



def combine_weight_kernel(tw_space, w_time_output, w_time_input, device):

    out_channels, output_stabilizer_size, in_channels, input_stabilizer_size,  kernel_size, kernel_size = tw_space.size()

    out_channels += 1
    in_channels += 1

    tw_shape = ((out_channels-1), output_stabilizer_size ,
                    (in_channels-1), input_stabilizer_size,  kernel_size* kernel_size)
    tw_space = tw_space.view(tw_shape)

    tw_space = tw_space.permute(1, 0, 3, 2, 4)

    tw_shape = (output_stabilizer_size , (out_channels-1),
                    input_stabilizer_size, (in_channels-1) * kernel_size* kernel_size)
    tw_space = tw_space.reshape(tw_shape)


    time_output_repeated = w_time_output.detach().repeat(output_stabilizer_size, 1, 1, 1)
    for i in range(output_stabilizer_size):
        time_output_repeated[i] = time_output_repeated[i].roll(i, dims=1)  # Example reorder logic

    # Concatenate along dim=1 (channel dimension)
    tw_space = torch.cat([time_output_repeated.to(device), tw_space.to(device)], dim=1)

    output_stabilizer_size, out_channels, input_stabilizer_size, rest = tw_space.size()
    tw_shape = (output_stabilizer_size, out_channels,
                    input_stabilizer_size * rest)
    tw_space = tw_space.reshape(tw_shape)

    time_input_expanded = w_time_input.unsqueeze(0).repeat(
    output_stabilizer_size,  1, 1  )

    tw = torch.cat([ time_input_expanded.to(device), tw_space.to(device)], dim=-1)

    return tw.contiguous()


class GroupLorentzConv2d(nn.Module):

    def __init__(
            self,
            input_stabilizer_size,
            output_stabilizer_size,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(GroupLorentzConv2d, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_stabilizer_size=input_stabilizer_size
        self.output_stabilizer_size=output_stabilizer_size
        self.manifold = manifold
        # space channel + time
        self.in_channels= in_channels
        # space channel * input + time
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)


        self.inds = self.make_transformation_indices()

        self.weight_space = Parameter(torch.Tensor(self.out_channels-1, self.in_channels-1, self.input_stabilizer_size, self.kernel_size[0] , self.kernel_size[1]), requires_grad=True)
        self.w_time_output = Parameter(torch.Tensor(
            1, 1, self.input_stabilizer_size, (self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]), requires_grad=True)
        self.w_time_input = Parameter(torch.Tensor(
            self.out_channels, 1), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels), requires_grad=True)
        else:
            self.bias = None
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.heads = nn.ModuleList([
           GroupLorentzFullyLinear(
            manifold,
            bias=bias,
            normalize=LFC_normalize
            ) for _ in range(self.output_stabilizer_size)
        ])


    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.kernel_size[1])

    def reset_parameters(self):
        # print(self.in_channels)
        stdv = math.sqrt(2.0 / ((self.in_channels-1) 
                                * self.kernel_size[0] * self.kernel_size[1]))

        # Initialize weight tensors
        self.weight_space.data.uniform_(-stdv, stdv)
        self.w_time_output.data.uniform_(-stdv, stdv)
        self.w_time_input.data.uniform_(-stdv, stdv)

        # Initialize bias terms if present

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)



    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        # x = (batch_size, input_stabliser, image-height, image-width, space out channels + 1 time out channels)
        # here the time channels is not one row but k-box

        bsz = x.shape[0]

        if self.input_stabilizer_size == 1:
            h, w, c = x.shape[1:4]
        else:
            g, h, w, c = x.shape[1:5]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        # print("x",x)
        if self.input_stabilizer_size == 1:
            x = x.permute(0, 3, 1, 2)
            # bsz, c, h, w

            # print("x (layer 1) shape", x.shape)
            patches = self.unfold(x)
            patches = patches.permute(0, 2, 1)
            # print("patches (layer 1) shape", patches.shape)
            patches_pre_kernel = self.extract_lorentz_patches(patches)

        else:

            x = self.manifold.lorentz_flatten_group_dimension(x)
            # bsz, h, w, (c-1)*9+1
            # print("x shape", x.shape)

            x = x.permute(0, 3, 1, 2)
            patches = self.unfold(x)
            patches = patches.permute(0, 2, 1)
            # print("patches (layer 1) shape", patches.shape)
            patches_pre_kernel = self.extract_lorentz_patches(patches)

            # print("patches", patches_pre_kernel.shape)

            # bsz, g, h, w, c
            # x = x.permute(1, 0, 4, 2, 3)
            # # print("x (layer 2) shape", x.shape)

            # patches_pre_kernel = []
            # for group_x in x:

            #     # print("group_shape", group_x.shape)
            #     patches = self.unfold(group_x)
            #     patches = patches.permute(0, 2, 1)
            #     # print("patches (layer 2) shape", patches.shape)
            #     patches_pre_kernel_single = self.extract_lorentz_patches(patches)
            #     patches_pre_kernel.append(patches_pre_kernel_single)

            # patches_pre_kernel = torch.stack(patches_pre_kernel, dim=0)  # shape: [B,  F]
            # # print("patches (3)", patches_pre_kernel.shape)
            # g, bsz, num_p, in_c = patches_pre_kernel.size()
            # patches_pre_kernel = patches_pre_kernel.permute(1,2,0,3)
            # #bsz, num_p, g, in_c

            # patches_pre_kernel = self.manifold.lorentz_flatten_group_patches(patches_pre_kernel)
            # print("patches (4)", patches_pre_kernel.shape)
            #!!!# hyperbolic concatenation
            # patches_pre_kernel = patches_pre_kernel.reshape(bsz, num_p, g * in_c)
            # print("patches (5)", patches_pre_kernel.shape)

            #can concateneate into a big ball???

        tw_space = trans_filter(self.weight_space, self.inds)

        weight_list = combine_weight_kernel(tw_space, self.w_time_output, self.w_time_input, self.device)
        # print("weight", weight_list[0].shape)


        outs = [head(patches_pre_kernel, weight) for head, weight in zip(self.heads, weight_list)]

         # Stack along a new “head” dimension: [B, P, num_heads, K, out_ch]
        out = torch.stack(outs, dim=0)

        # hyperbolic splittting!!!
        # print("out in conv2d",out.shape)

        # out = self.manifold.lorentz_split_patches(out, self.output_stabilizer_size)
        # print("out shape", out.shape)
        # bs, num, g, c
        # print(" output", out.shape)
        out = out.permute(1,0,2,3)
        out = out.view(bsz, self.output_stabilizer_size, h_out, w_out, self.out_channels)

        # print("final", out.shape)
        #(batch size, image_height, image_wedith, space channel + transformed space channel + time channels)

        return out

    def extract_lorentz_patches(self, patches):
        ## step 3： extract the time component from patches and treat them separately
        # Extracts the time coordinate and ensures it does not go below the hyperbolic manifold’s threshold.
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())
        # patches_time = (batch_size,  num of patches, kernel size (since it is last element and it is belong to time))
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))
       #  torch.Size([128, 256, 1])

        ## step 4: Extracts the remaining spatial components from patches.
        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        # torch.Size([128, 256, 9])


        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape)
        # torch.Size([128, 256, 9])

        ## step 5: Concatenates the rescaled time component and spatial components to maintain hyperbolic consistency
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)
        # patches = (batch_size, 1(time) + [channels(-1) * kernel_height * kernel_width], num_patches)

        return patches_pre_kernel






class LorentzP4MConvZ2(GroupLorentzConv2d):

    def __init__(self, *args, **kwargs):
        super(LorentzP4MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, *args, **kwargs)


class LorentzP4MConvP4M(GroupLorentzConv2d):

    def __init__(self, *args, **kwargs):
        super(LorentzP4MConvP4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)



class LorentzP4ConvZ2(GroupLorentzConv2d):

    def __init__(self, *args, **kwargs):
        super(LorentzP4ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, *args, **kwargs)


class LorentzP4ConvP4(GroupLorentzConv2d):

    def __init__(self, *args, **kwargs):
        super(LorentzP4ConvP4, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, *args, **kwargs)