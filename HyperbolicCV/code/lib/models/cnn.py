import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzConv2d, 
    LorentzBatchNorm1d,
    LorentzBatchNorm2d, 
    LorentzFullyConnected,
    LorentzMLR,
    LorentzReLU, 
    LorentzGlobalAvgPool2d
)
from groupy.gconv.pytorch_gconv.pooling import global_max_pooling
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M

equivariant_num = {
    "P4M": 8,
    "P4": 4
}

class CNN(nn.Module):
    def __init__(self, manifold: CustomLorentz = None,
                 img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False, eq_type = None):
        super(CNN, self).__init__()
        self.manifold = manifold  # Lorentz manifold
        self.embed_dim = embed_dim  # Set embedding dimension
        self.eq_type = eq_type
        self.img_dim = img_dim

        # **Using strided convolutions instead of MaxPool2d**
        self.conv1 = self.get_Conv2d(img_dim[0], 64, kernel_size=3, stride=2, padding=1)  # Stride 2 downsamples
        self.bn1 = self.get_BatchNorm2d(64)

        self.conv2 = self.get_Conv2d( 64, 128, kernel_size=3, stride=2, padding=1)  # Downsampling again
        self.bn2 = self.get_BatchNorm2d(128)

        self.conv3 = self.get_Conv2d( 128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = self.get_BatchNorm2d(256)

        self.conv4 = self.get_Conv2d( 256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = self.get_BatchNorm2d(512)

        # Lorentz Global Pooling (instead of max pooling)
        self.pooling = self._get_GlobalAveragePooling()

        # Fully Connected Layer
        if self.manifold is not None:
            self.fc1 = LorentzFullyConnected(self.manifold, in_features=512+1, out_features=self.embed_dim)
        elif self.eq_type is not None:
            self.fc1 = nn.Linear(512 * equivariant_num[self.eq_type], self.embed_dim)
        else:
            self.fc1 = nn.Linear(512, self.embed_dim)
        self.activation = self.get_Activation()

        # map the dimension to the num of class as final output logit [batch, num_class], 
        # but if we have decoder, we don't need to do this here, it will do it later in decoder side
        if remove_linear:
            self.predictor = None
        else:
            self.predictor = self._get_predictor(self.embed_dim, num_classes)
          


    def forward(self, x):
        #batch = 128
        # Input shape: [batch, 3, 32, 32]
        if self.manifold is not None:
            x = x.permute(0, 2, 3, 1)  # Convert to (batch, H, W, C) for Lorentz ops
            x = self.manifold.projx(F.pad(x, pad=(1, 0)))  # Lorentz projection. Adds an extra dimension of time at the start
        # Shape: [batch, 32, 32, 4]
   
        x = self.activation(self.bn1(self.conv1(x)))    # Strided Conv downsamples
        # Shape: [batch, 16, 16, 65]
        # equivariant: [batch, 64, 8, 16, 16]
        # print("shape 1:", x.shape)

        x = self.activation(self.bn2(self.conv2(x)))  
        # Shape: [batch, 8, 8, 129]
        # equivariant: [batch, 128, 8, 8, 8]
        # print("shape 2:", x.shape)

        x = self.activation(self.bn3(self.conv3(x)))  
        # Shape: [batch, 4, 4, 257]
        # equivariant: [batch, 256, 8, 4, 4]
        # print("shape 3:", x.shape)

        x = self.activation(self.bn4(self.conv4(x)))  
        # Shape: [batch, 2, 2, 513]
        # equivariant: [batch, 512, 8, 2, 2]
        # print("shape 4:", x.shape)

        x = self.pooling(x)  # Global Pooling in Lorentz space
        # Shape: [batch, 1, 1, 513]
        # equivariant: [batch, 512, 8, 1, 1]
        # print("shape 5:", x.shape)


        x = x.view(x.size(0), -1)  # Flatten for the fully connected (FC) layer.
        # Shape: [batch, 513]
        # eqivariant: [batch, 512 * 8=4096]
        # print("shape fc:", x.shape)

        x = self.fc1(x) # here it reduce a dimension
        # print("shape 6:", x.shape)
        # eqivariant: [batch, 512]

        x = self.activation(x)
        # Shape: [batch, 512]
        # eqivariant: [batch, 512]
        # print("shape 7:", x.shape)

        # x = F.dropout(x, training=self.training, p=0.1) #Uses dropout (10%) for regularization.
        #  It works by randomly setting some neuron activations to zero during training

        # add time dimension for the decoder (final classification)
        if self.manifold is not None:
            x = self.manifold.add_time(x)  # Ensure compatibility with LorentzMLR
        # Shape: [batch, 513]

        if self.predictor is not None:
            x = self.predictor(x)
            # Shape: [batch, 100]
 
        return x
    
    def _get_GlobalAveragePooling(self):

        if self.eq_type is not None:
            return global_max_pooling
        
        elif self.manifold is None:
            return nn.AdaptiveAvgPool2d((1, 1))

        elif type(self.manifold) is CustomLorentz:
            return LorentzGlobalAvgPool2d(self.manifold, keep_dim=True)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

    def _get_predictor(self, in_features, num_classes):
        if self.manifold is None:
            return nn.Linear(in_features, num_classes, bias=self.bias)

        elif type(self.manifold) is CustomLorentz:
            return LorentzMLR(self.manifold, in_features+1, num_classes)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

    def get_Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, LFC_normalize=False):
        if self.manifold is None:
            if self.eq_type is not None:
                if self.eq_type =="P4":
                    if in_channels == self.img_dim[0]:  # First layer operates on Z2 input
                        return P4ConvZ2(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                    else:  # Deeper layers operate on P4 feature maps
                        return P4ConvP4(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                if self.eq_type =="P4M":
                    if in_channels == self.img_dim[0]:  # First layer operates on Z2 input
                        return P4MConvZ2(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                    else:  # Deeper layers operate on P4 feature maps
                        return P4MConvP4M(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            else:
                return nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=padding, 
                    bias=bias
                )

        elif type(self.manifold) is CustomLorentz:
            return LorentzConv2d(
                manifold=self.manifold, 
                in_channels=in_channels+1, 
                out_channels=out_channels+1, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=bias, 
                LFC_normalize=LFC_normalize
            )

    def get_BatchNorm2d(self, num_channels):
        if self.eq_type is not None:
            if self.eq_type =="P4M":
                return nn.BatchNorm3d(num_channels)
            if self.eq_type =="P4":
                return nn.BatchNorm3d(num_channels)
        elif self.manifold is None:
            return nn.BatchNorm2d(num_channels)

        elif type(self.manifold) is CustomLorentz:
            return LorentzBatchNorm2d(manifold=self.manifold, num_channels=num_channels+1)

    def get_Activation(self):
        if self.manifold is None:
            return F.relu
        elif type(self.manifold) is CustomLorentz:
            return LorentzReLU(self.manifold)


def EUCLIDEAN_CNN(manifold= None,
                 img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False,type=None):
    model = CNN(img_dim=img_dim, embed_dim=embed_dim, num_classes=num_classes, remove_linear=remove_linear)
    return model

def Lorentz_CNN(manifold= None,
                 img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False,type=None):
    model = CNN(manifold,img_dim=img_dim, embed_dim=embed_dim, num_classes=num_classes, remove_linear=remove_linear)
    return model

def equivariant_CNN(manifold= None,
                 img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False,eq_type=None):
    model = CNN(img_dim=img_dim, embed_dim=embed_dim, num_classes=num_classes, remove_linear=remove_linear,eq_type=eq_type)
    return model