import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz


# fully connected layer (like a regular nn.Linear layer) but operates in a Lorentzian manifold.
class LorentzFullyConnected(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
        ):
        super(LorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        # might need to change!!!

        
        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        # This layer will perform a standard matrix multiplication during the forward pass

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):
        # Linear Transformation 
        # x = [batch_size, num_patches, channels]
        # nn.linear.weight = []
        # the important things is here where they linear the time component together
     
        # input: torch.Size([128, 256, 10]), [batch_size, num_patches, in_channels]
        # weight: [65, 10], output_channel, in_channel
        # weight control the output_channel, and ensure the in_channels is same as the input
        # output: torch.Size([128, 256, 65]), [batch_size, num_patches, out_features]
        x = self.weight(x)
  
        # torch.Size([128, 256, 65])
        # [batch_size, num_patches, out_features]
        # !!!! be careful, out_channel here is automatically +1 for time in beginning

        # Extract the Spatial Components
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)
        # [batch_size, num_patches, out_features - 1] after this operation.
        # The 1 here corresponds to the time component, which is discarded when extracting x_space.

        # so far not use yet, so can do it later!!!
        if self.normalize:
            #  normalization of the spatial components is applied:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            # Step 4: Mask and Normalize Spatial Components
            mask = square_norm <= 1e-10
            # This checks where the squared norm is too small (close to zero),
            # and applies a mask to handle these cases where normalization might lead to instability (division by zero).

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length
            # The spatial components are then scaled by the scale factor.

            # This formula is used to embed the time dimension in the Lorentzian space, 
            # where self.manifold.k is a constant related to the Lorentzian geometry. 
            # The small 1e-5 is added for numerical stability.
            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())
            # ensures that for the patches where the squared norm was very small, 
            # the time component is set to a default value (self.manifold.k.sqrt()),
            # avoiding any instability due to numerical errors.

            mask = mask==False
            # This inverts the mask, so that the mask is applied to the valid spatial components (i.e., not zero or too small)
            x_space = x_space * mask
            # This ensures that the spatial components are only retained where the mask is valid.
            
            x = torch.cat([x_time, x_space], dim=-1)
            #  the time component (x_time) and the spatial components (x_space) are concatenated along the last dimension to form the final output tensor x.
        else:
            x = self.manifold.add_time(x_space)
            # they recalculated the time base on x_space then added there.

        # out 3 torch.Size([128, 256, 65])
 
        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)
