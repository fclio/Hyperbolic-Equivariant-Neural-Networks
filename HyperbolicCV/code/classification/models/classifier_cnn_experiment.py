import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.lorentz.manifold import CustomLorentz
from lib.geoopt.manifolds.stereographic import PoincareBall

from lib.lorentz.layers import LorentzMLR
from lib.poincare.layers import UnidirectionalPoincareMLR

from lib.models import cnn_experiment as cnn_normal
from lib.models import cnn_experiment_small as cnn_small
from lib.models import cnn_experiment_big as cnn_big


def get_cnn_model_dict(cnn_size):
    if cnn_size == "small":
        return {
            "lorentz": cnn_small.Lorentz_CNN,
            "euclidean": cnn_small.EUCLIDEAN_CNN,
            "equivariant": cnn_small.equivariant_CNN,
            "lorentz_equivariant": cnn_small.Lorentz_equivariant_CNN,
        }
    elif cnn_size == "big":
        return {
            "lorentz": cnn_big.Lorentz_CNN,
            "euclidean": cnn_big.EUCLIDEAN_CNN,
            "equivariant": cnn_big.equivariant_CNN,
            "lorentz_equivariant": cnn_big.Lorentz_equivariant_CNN,
        }
    else:
        # fallback to normal cnn_experiment
        return {
            "lorentz": cnn_normal.Lorentz_CNN,
            "euclidean": cnn_normal.EUCLIDEAN_CNN,
            "equivariant": cnn_normal.equivariant_CNN,
            "lorentz_equivariant": cnn_normal.Lorentz_equivariant_CNN,
        }

# Define decoder mappings
EUCLIDEAN_DECODER = {'mlr': nn.Linear}
LORENTZ_DECODER = {'mlr': LorentzMLR}
POINCARE_DECODER = {'mlr': UnidirectionalPoincareMLR}

class CNNClassifier(nn.Module):
    def __init__(self, enc_type="lorentz", dec_type="lorentz", cnn_size ="",enc_kwargs={}, dec_kwargs={}):
        super(CNNClassifier, self).__init__()

        self.enc_type = enc_type
        self.dec_type = dec_type
        self.clip_r = dec_kwargs.get('clip_r', 1.0)  # Set default if not provided
        CNN_MODEL = get_cnn_model_dict(cnn_size)

        # Initialize Encoder (CNN)
        if enc_type not in CNN_MODEL:
            raise ValueError(f"Unknown encoder type: {enc_type}")
        # self.manifold = CustomLorentz(k=enc_kwargs['k'], learnable=enc_kwargs['learn_k']) if enc_type == "lorentz" else None
        # self.encoder = CNN_MODEL[enc_type](self.manifold, img_dim=enc_kwargs['img_dim'], embed_dim=enc_kwargs['embed_dim'], num_classes=enc_kwargs['num_classes'], remove_linear=True,eq_type=enc_kwargs['eq_type'])
        self.encoder = CNN_MODEL[enc_type](remove_linear=True, **enc_kwargs)
        
        self.enc_manifold = self.encoder.manifold if (enc_type == "lorentz" or enc_type == "lorentz_equivariant" ) else None

        # Initialize Decoder
        self.dec_manifold = None
        if dec_type == "euclidean" or dec_type == "equivariant" :
            self.decoder = EUCLIDEAN_DECODER[dec_kwargs['type']](
                dec_kwargs['embed_dim'], dec_kwargs['num_classes']
            )
        elif dec_type == "lorentz" or dec_type == "lorentz_equivariant":
            self.dec_manifold = CustomLorentz(
                k=dec_kwargs["k"], learnable=dec_kwargs.get('learn_k', False)
            )
            self.decoder = LORENTZ_DECODER[dec_kwargs['type']](
                self.dec_manifold, dec_kwargs['embed_dim'] + 1, dec_kwargs['num_classes']
            )
        elif dec_type == "poincare":
            self.dec_manifold = PoincareBall(
                c=dec_kwargs["k"], learnable=dec_kwargs.get('learn_k', False)
            )
            self.decoder = POINCARE_DECODER[dec_kwargs['type']](
                dec_kwargs['embed_dim'], dec_kwargs['num_classes'], True, self.dec_manifold
            )
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")

    def check_manifold(self, x):
        """Handles transformations between Euclidean, Lorentz, and Poincare manifolds"""
        if self.enc_type == "euclidean" and self.dec_type == "lorentz":
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm) * x
            x = self.dec_manifold.expmap0(F.pad(x, pad=(1, 0), value=0))
        elif self.enc_type == "euclidean" and self.dec_type == "poincare":
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm) * x
            x = self.dec_manifold.expmap0(x)
        elif self.enc_type == "lorentz" and self.dec_type == "euclidean":
            x = self.enc_manifold.logmap0(x)[..., 1:]
        elif self.enc_manifold and self.dec_manifold and self.enc_manifold.k != self.dec_manifold.k:
            x = self.dec_manifold.expmap0(self.enc_manifold.logmap0(x))
        return x

    def embed(self, x):
        """Extracts embeddings from the encoder"""
        x = self.encoder(x)
        return self.check_manifold(x)

    def forward(self, x):
        """Forward pass through encoder and decoder"""
        x = self.encoder(x)
        x = self.check_manifold(x)
        x = self.decoder(x)
        return x
