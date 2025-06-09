import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from lib.lorentz_equivariant.utils import get_group_channels
from lib.geoopt import ManifoldParameter
from lib.lorentz.manifold import CustomLorentz

device = "cuda" if torch.cuda.is_available() else "cpu"

class GroupLorentzBatchNorm(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, input_stabilizer_size:int):
        super(GroupLorentzBatchNorm, self).__init__()
        self.manifold = manifold
        num_features = (num_features-1)*input_stabilizer_size+1
        

        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=0.1):
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."

        beta = self.beta

        # print("training", self.training)
        if self.training:
            # Compute batch mean
            mean = self.manifold.centroid(x)
            # print("mean",mean.size())
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            # this is the problem with this !!!
            # print("x_T1 ", x_T)
            # dede
            x_T = self.manifold.transp0back(mean, x_T)
            # print("x_T2 ", x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.centroid(means, w=torch.tensor(((1-momentum), momentum), device=means.device))))
                self.running_var.copy_((1 - momentum)*self.running_var + momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output
    
class GroupLorentzBatchNorm2d(GroupLorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, input_stabilizer_size:int):
        super(GroupLorentzBatchNorm2d, self).__init__(manifold, num_channels, input_stabilizer_size)
        self.input_stabilizer_size = input_stabilizer_size

    def forward(self, x, momentum=0.1):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, g, h, w, c = x.shape
        print("before:", x.shape)
        
        x_group = self.manifold.lorentz_flatten_group_dimension(x)
  
        x_group = x_group.contiguous().view(bs, -1, g*(c-1)+1)  # Flatten groups into one batch
    
        x_group = super(GroupLorentzBatchNorm2d, self).forward(x_group, momentum)
        
        x = x_group.view(bs, h, w, g*(c-1)+1)

        x = self.manifold.lorentz_split_batch(x, g)

        return x
    
        # print("after:", x.shape)

        # x = x.permute(1, 0, 2, 3, 4)

        # list_x = []
        # for x_group in x:
        #     x_group = x_group.contiguous().view(bs, -1, c)  # Flatten groups into one batch
          
        #     x_group = super(GroupLorentzBatchNorm2d, self).forward(x_group, momentum)
            
        #     x_group = x_group.view(bs, h, w, c)

        #     list_x.append(x_group)
        
        # x = torch.stack(list_x, dim=0) 
        # x = x.permute(1, 0, 2, 3, 4)

        # return x
    
    def test(self, x, momentum):
        with open("error.json", "r") as f:
            loaded_list = json.load(f)

        # Convert back to a PyTorch tensor
        loaded_tensor = torch.tensor(loaded_list)
        x_original = super(GroupLorentzBatchNorm2d, self).forward(loaded_tensor, momentum)
        print("norm x oringinal", x_original)
        # should i have diff class for each patch? so some weight is not updated?
      
        dede


    def batchnorm_perVersion(self, x, momentum):
  
        x_space, x_time, x_space_original, x_original = get_group_channels(x, self.input_stabilizer_size)
        
        x_space_list = []

        x_original = super(GroupLorentzBatchNorm2d, self).forward(x_original, momentum)
        print("norm x oringinal", x_original)
        x_space_original = x_original.narrow(-1, 1, x_original.shape[-1] - 1)
        x_time_orginal = x_original.narrow(-1, 0, 1)
        x_space_list.append(x_space_original)


        for i in range(1, self.input_stabilizer_size):
            indices = torch.arange(i, x_space.size()[-1], step=self.input_stabilizer_size).to(device)
            x_space_versions = torch.index_select(x_space, dim=-1, index=indices).to(device)
            x_versions = torch.cat([x_time.to(device), x_space_versions.to(device)], dim=-1)
            print(f"norm x in {i}:", x_versions)
            x_new  = super(GroupLorentzBatchNorm2d, self).forward(x_versions, momentum)
            print(f"norm x new {i}:", x_new)
            
            if torch.isnan(x_new).any():
                with open("error.json", "w") as f:
                    json.dump(x_versions.tolist(), f)
                dedede
            
            x_space_list.append(x_new.narrow(-1, 1, x_new.shape[-1] - 1))

        

        stacked = torch.stack(x_space_list, dim=-1)
        result = stacked.view(x.size()[0], x.size()[1], x.size()[-1]-1)
        result = torch.cat([x_time_orginal.to(device), result.to(device)], dim=-1)
   
        return result