import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from lib.lorentz.manifold import CustomLorentz
import torch.nn.functional as F
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

    output_stabilizer_size, out_channels, input_stabilizer_size, rest = tw_space.size()
    tw_shape = (output_stabilizer_size * out_channels,
                    input_stabilizer_size * rest)
    tw = tw_space.reshape(tw_shape)

    time_input_expanded = w_time_input.detach().repeat(output_stabilizer_size, 1, 1)
    time_input_expanded = time_input_expanded.view(output_stabilizer_size*out_channels, 1)
    # Concatenate along dim=1 (channel dimension)
    tw_space = torch.cat([time_input_expanded.to(device), tw.to(device)], dim=1)

    tw_space = torch.cat([ w_time_output.to(device), tw_space.to(device)], dim=0)

    return tw.contiguous()




class GroupLorentzFullyConnected(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            lin_features,
            out_channels,
            in_channels,
            kernel_size,
            input_stabilizer_size,
            output_stabilizer_size,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
        ):
        super(GroupLorentzFullyConnected, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.manifold = manifold
        # assume here is the in_channel with time dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.ksize = kernel_size[0]
        self.normalize = normalize
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size


        self.weight_space = Parameter(torch.Tensor(
            self.out_channels-1, self.in_channels-1, self.input_stabilizer_size, self.kernel_size[0] , self.kernel_size[1]), requires_grad=True)
        self.weight_time_output = Parameter(torch.Tensor(
            1, self.input_stabilizer_size* (self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]+1), requires_grad=True)
        self.weight_time_input = Parameter(torch.Tensor(
            self.out_channels, 1), requires_grad=True)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.init_std = 0.02
        self.reset_parameters()

        self.inds = self.make_transformation_indices()
        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)
            
    def reset_parameters(self):
        nn.init.uniform_(self.weight_space, -self.init_std, self.init_std)
        nn.init.uniform_(self.weight_time_output, -self.init_std, self.init_std)
        nn.init.uniform_(self.weight_time_input, -self.init_std, self.init_std)

        # Initialize bias if it's present
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
             
    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, x):
        # x =[batch_size, num_patches, space channels + time channels (1)]
        # torch.Size([128, 256, 10])

        tw_space = trans_filter(self.weight_space, self.inds)
        tw = combine_weight_kernel(tw_space, self.weight_time_output, self.weight_time_input, self.device)
      
        # print(f"tw_space requires grad: {tw.requires_grad}")
        # print("tw shape", tw.shape)

        x = F.linear(x, tw, bias=None)
        # [batch_size, num_patches, out_features]
        # torch.Size([128, 256, 65])

        # if self.bias is not None:
        #     group_map = torch.arange(1,self.out_channels ).repeat_interleave(self.output_stabilizer_size).to(x.device)
        #     group_map = torch.cat([torch.tensor([0], device=x.device), group_map])
        #     bias = self.bias[group_map].view(1,  1, -1)  
        #     x = x + bias

        # Extract the Spatial Components
        # print(x.size())
        batch_size, num_patches, out_c = x.size()
        # print("x 1", x.shape)

        normalized_x = self.normalize_time(x)

        return normalized_x    # shape: [B, G, n, F]
    
    def normalize_time(self, x):
        x_space = x.narrow(-1, 1, x.shape[-1] - 1).to(self.device)
        x_time = x.narrow(-1, 0, 1).to(self.device)
 
        if self.normalize:
            x_time = self.extract_time(x, x_space)
            #  the time component (x_time) and the spatial components (x_space) are concatenated along the last dimension to form the final output tensor x.
        else:
            x_time = self.manifold.get_time(x_space)
        
        x = torch.cat([x_time.to(self.device), x_space.to(self.device)], dim=-1)

        return x
    def extract_time(self, x, x_space):
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
    
        return x_time
    
class GroupLorentzLinear(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_features,
            input_stabilizer_size,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
        ):
        super(GroupLorentzLinear, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.manifold = manifold
        # assume here is the in_channel with time dimension
        self.in_channels = in_channels
        self.out_channels = out_features
        self.bias = bias
        self.normalize = normalize
        self.input_stabilizer_size = input_stabilizer_size

        self.lin_features = (in_channels) * self.input_stabilizer_size
        
        self.weight = Parameter(torch.Tensor(
            self.out_channels, self.lin_features), requires_grad=True)
 
        
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels), requires_grad=True)
     

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)
            
    def reset_parameters(self):
        # Initialize weight_space with a uniform distribution
        nn.init.uniform_(self.weight, -self.init_std, self.init_std)
        # Initialize weight_time_input with a uniform distribution
  
        # Initialize bias if it's present
        if self.bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # x =[batch_size, num_patches, space channels + time channels (1)]

        x = F.linear(x, self.weight)

        # Extract the Spatial Components
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)
        x_time = x.narrow(-1, 0, 1)
      
        if self.normalize: 
            x_time = self.extract_time(x, x_space)
            #  the time component (x_time) and the spatial components (x_space) are concatenated along the last dimension to form the final output tensor x.
        else:
            x_time = self.manifold.get_time(x_space)
        
        x = torch.cat([x_time, x_space], dim=-1)

        return x

    def extract_time(self, x, x_space):
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
    
        return x_time
  