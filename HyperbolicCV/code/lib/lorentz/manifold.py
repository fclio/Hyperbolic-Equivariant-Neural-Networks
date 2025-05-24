import torch

from lib.geoopt import Lorentz
from lib.geoopt.manifolds.lorentz import math
import numpy as np


class CustomLorentz(Lorentz):
    def _init__(self, k=1.0, learnable=False):
        super(CustomLorentz, self).__init__(k=k, learnable=learnable)

    def sqdist(self, x, y, dim=-1):
        """ Squared Lorentzian distance, as defined in the paper 'Lorentzian Distance Learning for Hyperbolic Representation'"""
        return -2*self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)

    def add_time(self, space):
        """ Concatenates time component to given space component. """
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)
    def get_time(self, space):
        """ Concatenates time component to given space component. """
        time = self.calc_time(space)
        return time
    def calc_time(self, space):
        """ Calculates time component from given space component. """
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+self.k)

    def centroid(self, x, w=None, eps=1e-8):
        """ Centroid implementation. Adapted the code from Chen et al. (2022) """
        if w is not None:
            avg = w.matmul(x)
        else:
            avg = x.mean(dim=-2)

        denom = (-self.inner(avg, avg, keepdim=True))
        denom = denom.abs().clamp_min(eps).sqrt()

        centroid = torch.sqrt(self.k) * avg / denom

        return centroid

    def switch_man(self, x, manifold_in: Lorentz):
        """ Projection between Lorentz manifolds (e.g. change curvature) """
        x = manifold_in.logmap0(x)
        return self.expmap0(x)

    def pt_addition(self, x, y):
        """ Parallel transport addition proposed by Chami et al. (2019) """
        z = self.logmap0(y)
        z = self.transp0(x, z)

        return self.expmap(x, z)

    #################################################
    #       Reshaping operations
    #################################################
    def lorentz_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements flattening operation directly on the manifold. Based on Lorentz Direct Concatenation (Qu et al., 2022) """
        bs,h,w,c = x.shape
        # bs x H x W x C
        # 128, 16, 16, 65


        # hyperbolic input: h x w x c (time + space channnels)
        # output: bs, h x w x (c-1) + 1

        # each spatial position (h, w) is considered as an individual hyperbolic point (or “hyperbolic ball”),
        # and you have a total of h * w such hyperbolic points per example in the batch.


        time = x.narrow(-1, 0, 1).view(-1, h*w)
        print("time", time.shape)
        # time torch.Size([128, 16*16 = 256])

        space = x.narrow(-1, 1, x.shape[-1] - 1).flatten(start_dim=1) # concatenate all x_s
        print("space", space.shape)
        # [128, 16*16*(65-1) = 16384]

        time_rescaled = torch.sqrt(torch.sum(time**2, dim=-1, keepdim=True)+(((h*w)-1)/-self.k))
        print("time_rescaled", time_rescaled.shape)
        # torch.Size([128, 1])

        x = torch.cat([time_rescaled, space], dim=-1)


        return x

    def lorentz_flatten_group(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lorentz flattening over group, height, and width dimensions.
        Input shape: (bs, g, h, w, c)
        Output shape: (bs, 1 + (g * h * w) * (c - 1))
        """
        bs, g, h, w, c = x.shape

        # Extract time component (first dim of Lorentz vector)
        time = x[..., 0]  # (bs, g, h, w)

        # Flatten space component (rest of Lorentz vector)
        space = x[..., 1:]  # (bs, g, h, w, c-1)
        space = space.reshape(bs, -1)  # (bs, g * h * w * (c - 1))

        # Rescale time to preserve Lorentz norm
        time = time.reshape(bs, -1)  # (bs, g * h * w)
        time_squared = torch.sum(time**2, dim=-1, keepdim=True)  # (bs, 1)

        num_vectors = g * h * w
        time_rescaled = torch.sqrt(time_squared + ((num_vectors - 1) / -self.k))  # (bs, 1)

        # Concatenate new time with flattened space
        x_flat = torch.cat([time_rescaled, space], dim=-1)  # (bs, 1 + g * h * w * (c - 1))

        return x_flat

    # need to change and test my code:
    def lorentz_flatten_group_dimension(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lorentz flattening over group, height, and width dimensions.
        Input shape: (bs, g, h, w, c)
        Output shape: (bs, h, w, 1 + (g) * (c - 1))
        """
        bs, g, h, w, c = x.shape
        x = x.permute(0,2,3,1,4)

        # Extract time component (first dim of Lorentz vector)
        time = x[..., 0]  # (bs, h, w, g)

        # Flatten space component (rest of Lorentz vector)
        space = x[..., 1:]  # (bs,  h, w, g, c-1)
        space = space.reshape(bs, h, w, -1)  # (bs, h, w, g * (c - 1))

        # Rescale time to preserve Lorentz norm
        time = time.reshape(bs, h, w, -1)  # (bs, h, w, g)
        time_squared = torch.sum(time**2, dim=-1, keepdim=True)  # (bs, h, w, 1)

        num_vectors = g
        time_rescaled = torch.sqrt(time_squared + ((num_vectors - 1) / -self.k))  # (bs, h, w, 1)

        # Concatenate new time with flattened space
        x_flat = torch.cat([time_rescaled, space], dim=-1)  # (bs, h,w, 1 + g * (c - 1))

        return x_flat


    def lorentz_flatten_group_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lorentz flattening over group, height, and width dimensions.
        Input shape: (bs, g, h, w, c)
        Output shape: (bs, 1 + (g * h * w) * (c - 1))
        """
        bs, g, num_p, in_c = x.shape
        print("shape bs, g, num_p, in_c:", bs,g, num_p, in_c)
        # 128 64 4 577

        # Extract time component (first dim of Lorentz vector)
        time = x[..., 0]  # (bs, g)

        # Flatten space component (rest of Lorentz vector)
        space = x[..., 1:]  # (bs, num_p, g, c-1)
        space = space.reshape(bs, num_p, -1)  # (bs, num_p, g * (c - 1))

        # Rescale time to preserve Lorentz norm
        time = time.reshape(bs, num_p, -1)  # (bs, num_p, g)
        time_squared = torch.sum(time**2, dim=-1, keepdim=True)  # (bs, num_p, 1)
        print("time_squred shape", time_squared.shape)

        num_vectors = g
        time_rescaled = torch.sqrt(time_squared + ((num_vectors - 1) / -self.k))  # (bs,num_p, 1)

        # Concatenate new time with flattened space
        x_flat = torch.cat([time_rescaled, space], dim=-1)  # (bs, num_p, 1 + g * (c - 1))
        print("x_flat shape", x_flat.shape)

        return x_flat





    def lorentz_split_batch(self, x, g):
        """
        Lorentz Split for batched input of shape (bs, h, w, c),
        where c = 1 (time) + (g * space_dim_per_group)

        Args:
            x: np.ndarray of shape (bs, h, w, c)
            g: int, number of groups to split spatial dimensions
            K: float, curvature parameter (default=1.0)

        Returns:
            np.ndarray: shape (bs, h, w, g, space_dim_per_group + 1)
        """
        assert x.ndim == 4, "Expected input shape (bs, h, w, c)"
        t = x[..., :1]  # Time component
        s = x[..., 1:]  # Spatial components
        bs, h, w, c_minus1 = s.shape
        assert c_minus1 % g == 0, "Spatial dims must divide evenly into g groups"

        d = c_minus1 // g  # Dimensions per group

        s_split = s.reshape(bs, h, w, g, d)

        # print("s_split",s_split.shape)
        s_norm_sq = torch.sum(s_split ** 2, axis=-1, keepdims=True)
        new_t = torch.sqrt(s_norm_sq + 1.0 / self.k)  # Recalculate time component

        y = torch.cat([new_t, s_split], axis=-1)
        # print("y", y.shape)
        y = y.permute(0,3,1,2,4)


        return y  # Shape: (bs, h, w, g, d + 1)

    def lorentz_split_patches(self, x, g):
        """
        Lorentz Split for batched input of shape (bs, h, w, c),
        where c = 1 (time) + (g * space_dim_per_group)

        Args:
            x: np.ndarray of shape (bs, h, w, c)
            g: int, number of groups to split spatial dimensions
            K: float, curvature parameter (default=1.0)

        Returns:
            np.ndarray: shape (bs, h, w, g, space_dim_per_group + 1)
        """
        assert x.ndim == 4, "Expected input shape (bs, h, w, c)"
        t = x[..., :1]  # Time component
        s = x[..., 1:]  # Spatial components
        bs, n, c_minus1 = s.shape
        assert c_minus1 % g == 0, "Spatial dims must divide evenly into g groups"

        d = c_minus1 // g  # Dimensions per group

        s_split = s.reshape(bs, n , g, d)
        s_norm_sq = torch.sum(s_split ** 2, axis=-1, keepdims=True)
        new_t = torch.sqrt(s_norm_sq + 1.0 / self.k)  # Recalculate time component

        y = torch.cat([new_t, s_split], axis=-1)
    
        return y  



    def lorentz_reshape_img(self, x: torch.Tensor, img_dim) -> torch.Tensor:
        """ Implements reshaping a flat tensor to an image directly on the manifold. Based on Lorentz Direct Split (Qu et al., 2022) """
        space = x.narrow(-1, 1, x.shape[-1] - 1)
        space = space.view((-1, img_dim[0], img_dim[1], img_dim[2]-1))
        img = self.add_time(space)

        return img


    #################################################
    #       Activation functions
    #################################################
    def lorentz_relu(self, x: torch.Tensor, add_time: bool=True) -> torch.Tensor:
        """ Implements ReLU activation directly on the manifold. """
        return self.lorentz_activation(x, torch.relu, add_time)

    def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
        """ Implements activation directly on the manifold. """

        xs = x.narrow(-1, 1, x.shape[-1] - 1)
        # performs slicing along the last dimension (-1, meaning the channel dimension C)
        x = activation(xs)
        if add_time:
            x = self.add_time(x)
        return x

    def tangent_relu(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements ReLU activation in tangent space. """
        return self.expmap0(torch.relu(self.logmap0(x)))


