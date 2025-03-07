import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz
from groupy.gconv.make_gconv_indices import *
make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices}

def trans_filter(w, inds):
    
    #  self.weight: torch.Size([64, 1, 1, 3, 3])
    #  (out_channels, in_channels, input_stabilizer_size, kernel_size, kernel_size)
    #  self.inds: (8, 1, 3, 3, 3)
    #  (num_group_transformation, input_stabilizer_size, kernel_size, kernel_size, 3)
       
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    # indeces: (8 * 1 * 3 * 3, 3) = (72, 3)
    # Each row contains a (T, U, V) triplet that maps a pixel from the original kernel to a transformed version.

    #  step 2: this remaps the filter weights based on the transformation indices.
    # !!! here is a special index finding will become w(:,:, len(n-dimension_list matching follow select base on index))
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    # Extracts transformed filter pixels using (T, U, V) from w
    # w_indexed.shape = (64, 1, 72) (since 72 = 8 * 1 * 3 * 3)
    # (out_channels, in_channels, num_group_transformation * input_stabilizer_size * kernel_size * kernel_size)

    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    # torch.Size([64, 1, 8, 1, 3, 3])
    # (out_channels, in_channels, num_group_transformation, input_stabilizer_size, kernel_size, kernel_size)


    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    #  torch.Size([64, 8, 1, 1, 3, 3])
    # (out_channels, num_group_transformation, in_channels, input_stabilizer_size, kernel_size, kernel_size)
    

    # Ensures that the tensor is stored contiguously in memory
    # This improves efficiency for PyTorch computations
    return w_transformed.contiguous()


class GroupLorentzFullyConnected(nn.Module):
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
        super(GroupLorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        # might need to change here!!!
        # get this to reshape into kernel, kernel through self.weight.data
        # then apply the transformation
        # then flatten them into nn.Linear(self.in_features, self.out_features * num_group, bias=bias)
        # so i can pass all patches in as regular and get them out as self.out_features * num_group then .view(out_feature, num_group)
        # hint: some fold unfold operation
        # lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1
        
        self.weight = nn.Linear(self.in_features, self.out_features , bias=bias)
        # This layer will perform a standard matrix multiplication during the forward pass
        # liner.weight = torch.Size([65, 10])
        print("weight",self.weight.weight.shape)
        dedes
        self.init_std = 0.02
        self.reset_parameters()
        # start here after initiaztion!!!!
        self.inds = self.make_transformation_indices()
        self.weight.data.view(...,...,....)
        
        # ......
        

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)
    def make_transformation_indices(self):
        # to understand later!
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, x):
        # Linear Transformation 
        # [batch_size, num_patches, channels]
        # the important things is here where they linear the time component together
     
        # torch.Size([128, 256, 10])
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
