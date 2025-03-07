import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz


class GroupLorentzGlobalAvgPool2d(torch.nn.Module):
    """ Implementation of a Lorentz Global Average Pooling based on Lorentz centroid defintion. 
    """
    def __init__(self, manifold: CustomLorentz, keep_dim=False):
        super(GroupLorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, G, h, w, c = x.shape
        x = x.view(bs,G, -1, c)  # Reshape to (batch_size, H*W, channels) It flattens spatial dimensions (H×W) into one dimension.
        x = self.manifold.centroid(x)  # Computes the Lorentz centroid across all spatial positions.
        # output: [batch_size, channels]
        print("x pooling", x.shape)
        swsw
        if self.keep_dim:
            x = x.view(bs, G, 1, 1, c)

        return x
