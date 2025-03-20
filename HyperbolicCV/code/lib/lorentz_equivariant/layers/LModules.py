import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz_equivariant.utils import get_group_channels

class GroupLorentzGlobalAvgPool2d(torch.nn.Module):
    """ Implementation of a Lorentz Global Average Pooling based on Lorentz centroid defintion. 
    """
    def __init__(self, manifold: CustomLorentz, keep_dim=False):
        super(GroupLorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)  # Reshape to (batch_size, H*W, channels) It flattens spatial dimensions (H×W) into one dimension.
        x = self.manifold.centroid(x)  # Computes the Lorentz centroid across all spatial positions.
        # output: [batch_size, channels]
        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        return x

class GroupLorentzReLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. 
    """
    def __init__(self, manifold: CustomLorentz, input_stabilizer_size: int):
        super(GroupLorentzReLU, self).__init__()
        self.manifold = manifold
        self.input_stabilizer_size = input_stabilizer_size

    def forward(self, x, add_time: bool=True):
        return self.manifold.lorentz_relu(x)
    
        # # print(x.shape)
        # # print("activation in", x.shape)
        # x_space, x_time, x_space_original, x_original = get_group_channels(x,self.input_stabilizer_size)
        # # performs slicing along the last dimension (-1, meaning the channel dimension C)
        # x_space = torch.relu(x_space)

        # if add_time:
        #     x_time = self.manifold.get_time(x_space_original)
        #     x = torch.cat([x_time, x_space], dim=-1)
        # # print("activation out", x.shape)
        # return x
