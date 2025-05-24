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


import importlib

def import_equivariant_layers(exp_v):
    """
    Dynamically import equivariant layers module based on exp_v string or int.
    Returns a dict of imported classes.
    """
    base_module = "lib.lorentz_equivariant"

    # Handle different exp_v formats
    if exp_v == "":
        version_suffix = ""
    else:
        # Convert exp_v to string if not already
        version_suffix = f"_{exp_v}" if isinstance(exp_v, int) else f"_{exp_v}"

    try:
        LConv = importlib.import_module(f"{base_module}{version_suffix}.layers.LConv")
        LFC = importlib.import_module(f"{base_module}{version_suffix}.layers.LFC")
        LBnorm = importlib.import_module(f"{base_module}{version_suffix}.layers.LBnorm")
        LModules = importlib.import_module(f"{base_module}{version_suffix}.layers.LModules")
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import equivariant layers for exp_v={exp_v}: {e}")

    layers = {
        'LorentzP4MConvZ2': getattr(LConv, 'LorentzP4MConvZ2'),
        'LorentzP4MConvP4M': getattr(LConv, 'LorentzP4MConvP4M'),
        'LorentzP4ConvZ2': getattr(LConv, 'LorentzP4ConvZ2'),
        'LorentzP4ConvP4': getattr(LConv, 'LorentzP4ConvP4'),
        'GroupLorentzFullyConnected': getattr(LFC, 'GroupLorentzFullyConnected'),
        'GroupLorentzLinear': getattr(LFC, 'GroupLorentzLinear'),
        'GroupLorentzBatchNorm2d': getattr(LBnorm, 'GroupLorentzBatchNorm2d'),
        'GroupLorentzGlobalAvgPool2d': getattr(LModules, 'GroupLorentzGlobalAvgPool2d'),
        'GroupLorentzReLU': getattr(LModules, 'GroupLorentzReLU'),

    }

    return layers

class CNN(nn.Module):
    def __init__(self, manifold: CustomLorentz = None,
                 img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False, eq_type = None, exp_v=""):
        super(CNN, self).__init__()
        self.manifold = manifold  # Lorentz manifold
        self.embed_dim = embed_dim  # Set embedding dimension
        self.eq_type = eq_type
        self.img_dim = img_dim
        self.exp_v = exp_v

        self.get_import()

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
        final_in_channels = 512
        if self.manifold is not None:
            if self.eq_type is not None:
                self.fc1 = self.GroupLorentzLinear(self.manifold, in_channels=final_in_channels+1, out_features=self.embed_dim, input_stabilizer_size=equivariant_num[self.eq_type])
            else:
                self.fc1 = LorentzFullyConnected(self.manifold, in_features=final_in_channels+1, out_features=self.embed_dim)
        elif self.eq_type is not None:
            self.fc1 = nn.Linear(final_in_channels * equivariant_num[self.eq_type], self.embed_dim)
        else:
            self.fc1 = nn.Linear(final_in_channels, self.embed_dim)
        self.activation = self.get_Activation()
        self.activation_final = self.get_Activation(final=True)

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

        # x = self.conv1(x)
        # print("after 1",x.shape)

        # x = self.bn1(x)
        # print("after 2",x.shape)


        # x = self.activation(x)
        # print("after 3", x.shape)



        x = self.activation(self.bn1(self.conv1(x)))   # Strided Conv downsamples
        # print("after",x)
        # Shape: [batch, 16, 16, 65]
        # equivariant: [batch, 64, 8, 16, 16]
        # lorenz+equivairant: [batch, 16, 16, 64 * 8 +1] = [batch, 16, 16, 513]
        # lorenz+equivairant v2: [128, 8, 16, 16, 65]
        # print("layer 1", x.shape)
        # layer 1 torch.Size([128, 8, 16, 16, 65])
        # print("layer 1", x)

        x = self.activation(self.bn2(self.conv2(x)))
        # x = self.conv2(x)
        # print("layer 2, conv3", x)
        # x = self.bn2(x)
        # print("layer 2, bn3", x)
        # x = self.activation(x)
        # print("layer 2, act", x)
        # Shape: [batch, 8, 8, 129]
        # equivariant: [batch, 128, 8, 8, 8]
        # lorenz+equivairant: [batch, 8, 8, 128 * 8 +1] = [batch, 8, 8, 1025]
        # lorenz+equivairant v2: [128, 8, 8, 8, 129]
        # print("layer 2", x.shape)
        # print("layer 2", x)
        # there is one nan somewhere
        # sesw
        # x = self.conv3(x)
        # print("layer 3, conv3", x)
        # x = self.bn3(x)
        # # print("layer 3, bn3", x)
        # x = self.activation(x)
        # # print("layer 3, act", x)

        x = self.activation(self.bn3(self.conv3(x)))
        # Shape: [batch, 4, 4, 257]
        # equivariant: [batch, 256, 8, 4, 4]
        # lorenz+equivairant: [batch, 4, 4, 256 * 8 +1] = [batch, 16, 16, 2048]
        # lorenz+equivairant v2: [128, 8, 4, 4, 257]
        # print("shape 3:", x.shape)
        # print("layer 3", x)


        x = self.activation(self.bn4(self.conv4(x)))
        # Shape: [batch, 2, 2, 513]
        # equivariant: [batch, 512, 8, 2, 2]
        # lorenz+equivairant: [batch, 2, 2, 512 * 8 +1] = [batch, 16, 16, 4105]
        # lorenz+equivairant v2: [128, 8, 2, 2, 513]
        # print("shape 4:", x.shape)
        # print("layer 4", x)




        x = self.pooling(x)  # Global Pooling in Lorentz space
        # Shape: [batch, 1, 1, 513]
        # equivariant: [batch, 512, 8, 1, 1]
        # lorenz+equivairant: [batch, 2, 2, 512 * 8 +1] = [batch, 1, 1, 4105]
        # lorenz+equivairant v2: [128, 8, 1, 1, 513]
        # print("shape 5:", x.shape)
        # print("layer 5", x)

        # Concatenation for group equivariant here!!!
        # only flattening for lorentz !!!
        if (self.manifold is not None) and (self.eq_type is not None):
            if self.exp_v == "v2":
                x = self.manifold.lorentz_flatten_group(x)
                # print("flattern", x.shape)
            else:
                x = x.reshape(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)  # Flatten for the fully connected (FC) layer.
        # Shape: [batch, 513]
        # eqivariant: [batch, 512 * 8=4096]
        # lorenz+equivairant: [batch, 512 * 8 + 1=4097]
        # lorenz+equivairant v2: [128, 512 * 8 +1 = 4104]
        # print("layer 6", x.shape)


        x = self.fc1(x) # here it reduce a dimension
        # Shape: [batch, 512]
        # print("shape 6:", x.shape)
        # eqivariant: [batch, 512]
        # lorenz+equivairant: [batch, 512]
        # lorenz+equivairant: [batch, 512]
        # print("layer 7", x.shape)

        x = self.activation_final(x)
        # Shape: [batch, 512]
        # eqivariant: [batch, 512]
        # print("shape 8:", x.shape)
        # print("layer 8", x)

        # x = F.dropout(x, training=self.training, p=0.1) #Uses dropout (10%) for regularization.
        #  It works by randomly setting some neuron activations to zero during training

        # add time dimension for the decoder (final classification)
        if self.manifold is not None:
            x = self.manifold.add_time(x)  # Ensure compatibility with LorentzMLR
        # Shape: [batch, 513]
        # print("layer 9", x.shape)
        if self.predictor is not None:
            x = self.predictor(x)
            # Shape: [batch, 100]
        # print("final", x.shape)
        dede

        return x


    def get_import(self):
        layers = import_equivariant_layers(self.exp_v)
        # Assign imported classes to class attributes
        self.LorentzP4MConvZ2 = layers['LorentzP4MConvZ2']
        self.LorentzP4MConvP4M = layers['LorentzP4MConvP4M']
        self.LorentzP4ConvZ2 = layers['LorentzP4ConvZ2']
        self.LorentzP4ConvP4 = layers['LorentzP4ConvP4']
        self.GroupLorentzFullyConnected = layers['GroupLorentzFullyConnected']
        self.GroupLorentzLinear = layers['GroupLorentzLinear']
        self.GroupLorentzBatchNorm2d = layers['GroupLorentzBatchNorm2d']
        self.GroupLorentzGlobalAvgPool2d = layers['GroupLorentzGlobalAvgPool2d']
        self.GroupLorentzReLU = layers['GroupLorentzReLU']


    def _get_GlobalAveragePooling(self):

        if self.manifold is None:
            if self.eq_type is not None:
                return global_max_pooling
            else:
                return nn.AdaptiveAvgPool2d((1, 1))

        elif type(self.manifold) is CustomLorentz:
            if self.eq_type is not None:
                return self.GroupLorentzGlobalAvgPool2d(self.manifold, keep_dim=True)
            else:
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
            if self.eq_type is not None:
                if self.eq_type =="P4":
                    if in_channels == self.img_dim[0]:  # First layer operates on Z2 input
                        return self.LorentzP4ConvZ2(manifold=self.manifold,
                                in_channels=in_channels+1,
                                out_channels=out_channels+1,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                                LFC_normalize=LFC_normalize)
                    else:  # Deeper layers operate on P4 feature maps
                        return self.LorentzP4ConvP4(manifold=self.manifold,
                                in_channels=in_channels+1,
                                out_channels=out_channels+1,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                                LFC_normalize=LFC_normalize)
                if self.eq_type =="P4M":
                    if in_channels == self.img_dim[0]:  # First layer operates on Z2 input
                        return self.LorentzP4MConvZ2(manifold=self.manifold,
                                in_channels=in_channels+1,
                                out_channels=out_channels+1,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                                LFC_normalize=LFC_normalize)
                    else:  # Deeper layers operate on P4 feature maps
                        return self.LorentzP4MConvP4M(manifold=self.manifold,
                                in_channels=in_channels+1,
                                out_channels=out_channels+1,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                                LFC_normalize=LFC_normalize)
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

        if self.manifold is None:
            if self.eq_type is not None:
                if self.eq_type =="P4M":
                    return nn.BatchNorm3d(num_channels)
                if self.eq_type =="P4":
                    return nn.BatchNorm3d(num_channels)
            else:
             return nn.BatchNorm2d(num_channels)

        elif type(self.manifold) is CustomLorentz:

            if self.eq_type is not None:
                return self.GroupLorentzBatchNorm2d(manifold=self.manifold, num_channels=num_channels+1,input_stabilizer_size= equivariant_num[self.eq_type])
            else:
                return LorentzBatchNorm2d(manifold=self.manifold, num_channels=num_channels+1)

    def get_Activation(self, final=False):
        if self.manifold is None:
            return F.relu
        elif type(self.manifold) is CustomLorentz:
            if final == False and self.eq_type is not None:
                return self.GroupLorentzReLU(manifold=self.manifold,input_stabilizer_size= equivariant_num[self.eq_type])
            else:
                return LorentzReLU(self.manifold)


def EUCLIDEAN_CNN(img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False):
    model = CNN(img_dim=img_dim, embed_dim=embed_dim, num_classes=num_classes, remove_linear=remove_linear)
    return model

def Lorentz_CNN(k=1, learn_k=False, img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False):
    manifold = CustomLorentz(k=k, learnable=learn_k)
    model = CNN(manifold,img_dim=img_dim, embed_dim=embed_dim, num_classes=num_classes, remove_linear=remove_linear)
    return model

def equivariant_CNN(img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False,eq_type=None):
    model = CNN(img_dim=img_dim, embed_dim=embed_dim, num_classes=num_classes, remove_linear=remove_linear,eq_type=eq_type)
    return model

def Lorentz_equivariant_CNN(k=1, learn_k=False, img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, remove_linear=False,eq_type=None, exp_v=""):
    manifold = CustomLorentz(k=k, learnable=learn_k)
    model = CNN(manifold,img_dim=img_dim, embed_dim=embed_dim, num_classes=num_classes, remove_linear=remove_linear,eq_type=eq_type, exp_v=exp_v)
    return model