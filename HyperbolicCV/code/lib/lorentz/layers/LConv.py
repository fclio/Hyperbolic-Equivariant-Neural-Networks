import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzFullyConnected

class LorentzConv1d(nn.Module):
    """ Implements a fully hyperbolic 1D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, bias: Same as nn.Conv1d
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1
        
        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt()) 

        patches = x.unfold(1, self.kernel_size, self.stride)
        # Lorentz direct concatenation of features within patches
        patches_time = patches.narrow(2, 0, 1)
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=(-2,-1), keepdim=True) - ((self.kernel_size - 1) * self.manifold.k))
        patches_time_rescaled = patches_time_rescaled.view(bsz, patches.shape[1], -1)

        patches_space = patches.narrow(2, 1, patches.shape[2]-1).reshape(bsz, patches.shape[1], -1)
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        # it use the LorentzFullyConnected forward here
        out = self.linearized_kernel(patches_pre_kernel)

        return out


class LorentzConv2d(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
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
        super(LorentzConv2d, self).__init__()

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
        #  Number of elements in the kernel (H * W).
        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]
        # calculates the total number of features after applying a 2D convolution operation
        # since we normally added +1 before passed to the conv2d for time component, (in_channels - 1) = Excludes the time component (the first channel) 
        # input channel * all elements in kernel = all kernel element in each channel.
        # +1: The time component in Lorentz space.
        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1
        # print("parameter: ", self.in_channels, self.kernel_size, self.out_channels)
        # in_channels: 2
        # kernel_size: (3, 3)
        # out_channels: 65
        # lin_features: 10


        # Instead of using a standard linear layer, this uses LorentzFullyConnected to preserve hyperbolic properties.
        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )


        # Extracts sliding windows (patches) from the input tensor, similar to what a convolution does.
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    # we are directly modifying the weights of the nn.Linear layer inside LorentzFullyConnected, which use later in forward
    # need to modify the connected LFC initation part if need to add the equivariant part!!!
    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        # print("hyperbolic weight:", a)
        # out_channel, lin_channel
        # torch.Size([65, 10])
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        ## step 1: Computes the output height and width after convolution without account for any lorenz point
        # x = (batch_size, image height, image width, space channels + time channels)
        # torch.Size([128, 32, 32, 2])
        # print("x", x.shape)
       
        # !!! remember here for the channels, it is normal_channels + 1 (time)
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # 16

        ## step 2: Extracting Patches
        # extracting local patches from the input tensor and reshaping them to prepare for further processing, like applying a linear layer or convolutional kernel.
        x = x.permute(0, 3, 1, 2)
        # x = (batch_size, channels, height, width)
        # torch.Size([128, 2, 32, 32])

        # used to extract sliding local blocks (or patches) from the input tensor
        patches = self.unfold(x)  
   
        # patches = (batch_size, channels(+1 time) * kernel_height * kernel_width, num_patches)
        # The number of elements per patch /each channels with each kernerl element: 2 * 3* 3
        # num of patches that can be extracted from each image = ((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        # patches = torch.Size([128, 18, 256])
        # therefore, These patches contain both spatial and time information.

        patches = patches.permute(0, 2, 1)
        # patches = (batch_size,  windows, channels * elements/window)
        # patches = torch.Size([128, 256, 18])
        
        #  each spatial position (k_size, k_size) in kernel size * kernel size is considered as an individual hyperbolic point (or “hyperbolic ball”),
        #  and you have a total of k_size * k_size such hyperbolic points per example in the batch.

        ## step 3： extract the time component from patches and treat them separately
        # Apply Lorentz Concatenation (Qu et al., 2022)
        # Now we have flattened patches with multiple time elements 
        # -> fix the concatenation to perform Lorentz direct concatenation by Qu et al. (2022)
        # Extracts the time coordinate and ensures it does not go below the hyperbolic manifold’s threshold.
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())  
        # 1. narrow: would extract the first kernel_len from each window (slice the last dimension from index 0 to 9).
        # 1. [128, 256, 9].
        # 2. clamp: ensures that any value in the tensor smaller than self.manifold.k.sqrt() (e.g., 0.1) will be replaced by 0.1. # Fix zero (origin) padding
        # patches_time = (batch_size,  num of patches, kernel size (since it is last element and it is belong to time))

        # step 2: Computes the rescaled time component using the Lorentz metric.
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))
        #  summing the squared values of the time component for each patch.
        #  Adjusts for the curvature of the hyperbolic manifold by subtracting ((self.kernel_len - 1) * self.manifold.k).
        # Takes the square root of the result to normalize the time components and ensure they remain in accordance with the Lorentz model.patches_time = (batch_size,  windows, 1)
        #  torch.Size([128, 256, 1])



        ## step 4: Extracts the remaining spatial components from patches.
        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        # torch.Size([128, 256, 9])
        # No, the reshaping itself does not change the data
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) 
        # No need, but seems to improve runtime??
        # torch.Size([128, 256, 9])



        ## step 5: Concatenates the rescaled time component and spatial components to maintain hyperbolic consistency
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)
        # patches = (batch_size, num_patches, 1(time) + [channels(-1) * kernel_height * kernel_width])
        # pathes: torch.Size([128, 256, 10])

        # print("patches", patches.shape)
        # print("patches_pre_kernel", patches_pre_kernel.shape)

        # step 6: Apply Lorentz Fully Connected Layer
        out = self.linearized_kernel(patches_pre_kernel)

        # torch.Size([128, 256, 65])
     
        out = out.view(bsz, h_out, w_out, self.out_channels)
        # Passes patches through the LorentzFullyConnected layer.
        # Reshapes the output to match the expected 2D convolution output.
        # here the self.out_channels is also added 1 for time (maybe for later case)
        # torch.Size([128, 16, 16, 65])

        # here why using this shape? why having out_channel at the end? for hyperbolic?
        
        return out

class LorentzConvTranspose2d(nn.Module):
    """ Implements a fully hyperbolic 2D transposed convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias: Same as nn.ConvTranspose2d
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            bias=True,
            LFC_normalize=False
        ):
        super(LorentzConvTranspose2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding

        padding_implicit = [0,0]
        padding_implicit[0] = kernel_size - self.padding[0] - 1 # Ensure padding > kernel_size
        padding_implicit[1] = kernel_size - self.padding[1] - 1 # Ensure padding > kernel_size

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1,1,1)),(1,1,1,1)), requires_grad=False)

        self.conv = LorentzConv2d(
            manifold=manifold, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding_implicit, 
            bias=bias, 
            LFC_normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        if self.stride[0] > 1 or self.stride[1] > 1:
            # Insert hyperbolic origin vectors between features
            x = x.permute(0,3,1,2)
            # -> Insert zero vectors
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,3,1)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = self.conv(x)

        if self.output_padding[0] > 0 or self.output_padding[1] > 0:
            x = F.pad(x, pad=(0, self.output_padding[1], 0, self.output_padding[0])) # Pad one side of each dimension (bottom+right) (see PyTorch documentation)
            x[..., 0].clamp_(min=self.manifold.k.sqrt()) # Fix origin padding

        return x
