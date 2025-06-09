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
        bs, g, h, w, c = x.shape

        x = x.permute(1, 0, 2, 3, 4)
        
        list_x = []
        for x_group in x:
            x_group = x_group.view(bs, -1, c)  # Reshape to (batch_size, H*W, channels) It flattens spatial dimensions (H×W) into one dimension.

            x_group = self.manifold.centroid(x_group)  # Computes the Lorentz centroid across all spatial positions.
            # output: [batch_size, channels]
            if self.keep_dim:
                x_group = x_group.view(bs, 1, 1, c)
            list_x.append(x_group)
            
        x = torch.stack(list_x, dim=0) 

        x = x.permute(1, 0, 2, 3, 4)

        return x

        # x = self.manifold.lorentz_flatten_group_dimension(x)

        # x = x.view(bs, -1, g*(c-1)+1)  # Reshape to (batch_size, H*W, channels) It flattens spatial dimensions (H×W) into one dimension.
        # x = self.manifold.centroid(x)  # Computes the Lorentz centroid across all spatial positions.
        # # output: [batch_size, channels]
        # if self.keep_dim:
        #     x = x.view(bs, 1, 1,  g*(c-1)+1)
        
        # x = self.manifold.lorentz_split_batch(x, g)

        # return x

class GroupLorentzReLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. 
    """
    def __init__(self, manifold: CustomLorentz, input_stabilizer_size: int):
        super(GroupLorentzReLU, self).__init__()
        self.manifold = manifold
        self.input_stabilizer_size = input_stabilizer_size

    def forward(self, x, add_time: bool=True):

        # bs, g, h, w, c = x.shape

        # x = self.manifold.lorentz_flatten_group_dimension(x)
        # print(x.shape)

        x = self.manifold.lorentz_relu(x) 

        x = self.manifold.lorentz_split_batch(x, self.input_stabilizer_size)

        return x
    
        
        
        # x = x.permute(1, 0, 2, 3, 4)

        # list_x = [self.manifold.lorentz_relu(x_group) for x_group in x]
        # x = torch.stack(list_x, dim=0) 

        # x = x.permute(1, 0, 2, 3, 4)
        # return x


