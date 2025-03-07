import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt import ManifoldParameter
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzConv2d, LorentzBatchNorm1d, LorentzBatchNorm

class GroupLorentzBatchNorm2d(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int):
        super(GroupLorentzBatchNorm2d, self).__init__(manifold, num_channels)

    def forward(self, x, momentum=0.1):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, num_groups, h, w, c = x.shape
        # print("norm:", x.shape)
        # torch.Size([128, 8, 16, 16, 65])
        # Reorganize x to make it in the form [num_groups * batch_size, H, W, C]
        x = x.permute(1, 0, 2, 3, 4)  # [num_groups, batch_size, H, W, C]
        x = x.contiguous().view(num_groups * bs, -1, c)  # Flatten groups into one batch
        # torch.Size([1024, 256, 65])

        # Apply Lorentz Batch Normalization
        x = super(GroupLorentzBatchNorm2d, self).forward(x, momentum)

        # Reshape back to original shape: [batch_size, num_groups, H, W, C]
        x = x.view(num_groups, bs, h, w, c)
        x = x.permute(1, 0, 2, 3, 4)  # Restore shape to [batch_size, num_groups, H, W, C]
        # print("norm 2:", x.shape)
        # torch.Size([128, 8, 16, 16, 65])
        # dsw
        return x
