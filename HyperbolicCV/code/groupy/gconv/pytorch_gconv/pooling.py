import torch.nn.functional as F


def plane_group_spatial_max_pooling(x, ksize, stride=None, pad=0):
    xs = x.size()
    x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
    x = F.max_pool2d(input=x, kernel_size=ksize, stride=stride, padding=pad)
    x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
    return x
import torch.nn.functional as F
import torch.nn as nn

def global_max_pooling(x):
    """
    Performs global max pooling over the spatial dimensions while maintaining 
    group structure, ensuring an xput size of (1,1).
    
    Args:
        x (Tensor): Input tensor of shape [B, G, C, H, W].
        
    Returns:
        Tensor: Pooled tensor with shape [B, G, C, 1, 1].
    """
    xs = x.size()
    x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])  # Flatten group structure
    x = F.adaptive_max_pool2d(x, (1, 1))  # Global max pooling to (1,1)
    x = x.view(xs[0], xs[1], xs[2], 1, 1)  # Reshape back

    return x

def global_average_pooling(x):
    xs = x.size()

    x = x.view(xs[0], xs[1]*xs[2], xs[3], xs[4])
    x = F.avg_pool2d(x, 4)
    x = x.view(xs[0], xs[1], xs[2], 1, 1)  # Reshape back
    return x