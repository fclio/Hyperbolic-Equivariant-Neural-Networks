import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzFullyConnected
from groupy.gconv.make_gconv_indices import *

make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices}

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
        self.input_stabilizer_size=input_stabilizer_size
        self.output_stabilizer_size=output_stabilizer_size
        self.manifold = manifold
        self.in_channels = in_channels
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

        # here need to reconsider for group!!!
        # for nn.linear, the kernel is also flatten ad contained in the input as linear, not as sliding windows
        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]
        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        # Instead of using a standard linear layer, this uses LorentzFullyConnected to preserve hyperbolic properties.
        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )


        # Extracts sliding windows (patches) from the input tensor, similar to what a convolution does.
        # !!! here for each group, there should be diff patches, so apply unfold for each to get diff patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

        self.inds = self.make_transformation_indices()
    def make_transformation_indices(self):
        # to understand later!
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.kernel_size[1])

    # where and when to update? do i need to keep the weight for each as well? 
    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        ## step 1: Computes the output height and width after convolution without account for any lorenz point
        # x = (batch_size, height, width, channels)
        # !!! remember here for the channels, it is normal_channels + 1 (time)
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        ## step 2: Extracting Patches
    	
        # extracting local patches from the input tensor and reshaping them to prepare for further processing, like applying a linear layer or convolutional kernel.
        xs = x.size()
        print("x in",x.shape)
        #  for 1-8: torch.Size([128, 32, 32, 2])
        # for 8-8:  torch.Size([128, 8, 16, 16, 65])

        # for 1-8: torch.Size([128, 2, 32, 32])
        # for 8-8: torch.Size([128, 520, 16, 16])
        # x = x.permute(0, 3, 1, 2)
        # x = (batch_size, channels, height, width)

        # torch.Size([128, 2, 32, 32])
        # used to extract sliding local blocks (or patches) from the input tensor
        # !!!!problem might be with unfold, for unfold it might need to applied to individual group instead of combine them like this
        # in equivairant as input change size the kernel also change size as well, but here the kernel size stay  the same
        x = x.view(xs[0], self.in_channels*self.input_stabilizer_size, xs[-3],xs[-2])
        print("x out",x.shape)
        patches = self.unfold(x)
        # patches = (batch_size, channels(+1 time) * kernel_height * kernel_width, num_patches)

        patches = patches.permute(0, 2, 1)
        _, num_patches , _ = patches.size()
        # patches = (batch_size,  windows, channels * elements/window)
        # torch.Size([128, 64, 4680])
        print("patches 1: ", patches.shape)
        if self.input_stabilizer_size == 1:
            patches = patches.view(bsz, num_patches, self.in_channels, self.input_stabilizer_size, self.kernel_size[0], self.kernel_size[1])
            patches = patches.repeat(1, 1, 1, self.output_stabilizer_size, 1, 1)
            print("patches 3: ", patches.shape)
        else:
            patches = patches.view(bsz, num_patches, self.in_channels, self.output_stabilizer_size, self.kernel_size[0], self.kernel_size[1])
            print("patches 2: ", patches.shape)
            #  torch.Size([128, 64, 65, 8, 3, 3])

        # patches = patches.repeat(1, 1, 1, self.input_stabilizer_size, 1, 1)
        # print("patches 3: ", patches.shape)

        patches_transformed = self.trans_filter_patches(patches, self.inds)

        # Initialize an empty list to store the results
        out_patches = []

        # Iterate through each patch and apply the linear layer
        for patches in patches_transformed:  # Iterate over num_patches
            # patches = (batch_size,  windows, channels * elements/window)
            #  torch.Size([128, 64, 4680])
            print("patches_transformed", patches.shape)
            patches_pre_kernel = self.extract_lorentz_patches(patches)

            out_patch = self.linearized_kernel(patches_pre_kernel)  # Apply the linear layer to each patch
        
            out_patches.append(out_patch)

        # Step 4: Stack the results into a single tensor
        out = torch.stack(out_patches, dim=1)  # Stack along the num_patches dimension

        out = out.view(bsz, self.output_stabilizer_size, h_out, w_out,self.out_channels )
        print("final", out.shape)
   
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
    
    def trans_filter_patches(self, patches, inds):
        """
        Transform patches using transformation indices `inds`.
        This function applies the transformation directly to the patches
        instead of the kernel weights.
        """
        # Indices reshape: Flatten the indices to apply to patches
        inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
        
        # Extract patches using the indices (mapping each patch to its transformed version)
        patches_indexed = patches[:, :, :, inds_reshape[:, 0], inds_reshape[:, 1], inds_reshape[:, 2]]
        
        # print("patches 4: ", patches_indexed.shape)
        # Reshape the indexed patches to match the output format
        patches_indexed = patches_indexed.reshape(
            patches_indexed.size()[0], patches_indexed.size()[1] , patches_indexed.size()[2] * inds.shape[1],
              inds.shape[0],  inds.shape[2] * inds.shape[3]
        )
        # print("patches 5: ", patches_indexed.shape)
        patches_indexed = patches_indexed.permute(3, 0, 1, 2, 4)
        # print("patches 6: ", patches_indexed.shape)
        patches_indexed = patches_indexed.reshape(
            patches_indexed.size()[0], patches_indexed.size()[1] , patches_indexed.size()[2],  patches_indexed.size()[3] *patches_indexed.size()[4] 
        )
        # print("patches 7: ", patches_indexed.shape)
        # Return the transformed patches
        return patches_indexed.contiguous()



    
class LorentzP4MConvZ2(GroupLorentzConv2d):

    def __init__(self, *args, **kwargs):
        super(LorentzP4MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, *args, **kwargs)


class LorentzP4MConvP4M(GroupLorentzConv2d):

    def __init__(self, *args, **kwargs):
        super(LorentzP4MConvP4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)