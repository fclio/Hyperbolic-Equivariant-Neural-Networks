import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
import json
from lib.lorentz.layers import (
    LorentzConv2d,
    LorentzBatchNorm1d,
    LorentzBatchNorm2d,
    LorentzFullyConnected,
    LorentzMLR,
    LorentzReLU,
    LorentzGlobalAvgPool2d
)

from lib.utils.equivariant_test.visualization import *
from groupy.gconv.pytorch_gconv.pooling import global_max_pooling
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
import torch.nn as nn
import importlib



class JSONLogger:
    def __init__(self, save_path):
        self.log = []
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def log_entry(self, message):
        print(message)
        self.log.append(message)

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.log, f, indent=4)


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF



class EquivarianceTester:
    def __init__(self, model, device, exp_v = "", model_type="", save_path="./equivariance_vis",eq_type="P4"):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.exp_v = exp_v
        self._load_layers()
        self.eq_type = eq_type
        self.save_path = save_path
        log_file=f"equivariance_log_{exp_v}.json"
        self.logger = JSONLogger(os.path.join(save_path, log_file))
        self.layer_types = (
            nn.Conv2d, LorentzConv2d, self.LorentzP4ConvZ2, self.LorentzP4ConvP4, self.GroupLorentzReLU, self.GroupLorentzGlobalAvgPool2d, P4ConvZ2, P4ConvP4,
            self.GroupLorentzBatchNorm, P4MConvZ2, P4MConvP4M, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d
        )
        os.makedirs(self.save_path, exist_ok=True)


    def _load_layers(self):
        # Determine module path based on self.exp_v
        if self.exp_v == '':
            base_path = 'lib.lorentz_equivariant'
        else:
            base_path = f'lib.lorentz_equivariant_{self.exp_v}'

        # Dynamically import LConv and LFC modules
        lconv_module = importlib.import_module(f'{base_path}.layers.LConv')
        lfc_module = importlib.import_module(f'{base_path}.layers.LFC')
        LModules_module = importlib.import_module(f'{base_path}.layers.LModules')
        LBnorm_module = importlib.import_module(f'{base_path}.layers.LBnorm')

        # Assign layer classes to instance attributes
        self.LorentzP4MConvZ2 = lconv_module.LorentzP4MConvZ2
        self.LorentzP4MConvP4M = lconv_module.LorentzP4MConvP4M
        self.LorentzP4ConvZ2 = lconv_module.LorentzP4ConvZ2
        self.LorentzP4ConvP4 = lconv_module.LorentzP4ConvP4
        self.GroupLorentzFullyConnected = lfc_module.GroupLorentzFullyConnected
        self.GroupLorentzLinear = lfc_module.GroupLorentzLinear
        self.GroupLorentzBatchNorm= LBnorm_module.GroupLorentzBatchNorm
        self.GroupLorentzGlobalAvgPool2d = LModules_module.GroupLorentzGlobalAvgPool2d
        self.GroupLorentzReLU = LModules_module.GroupLorentzReLU


    def apply_transformation(self, tensor, angle=0, flip=None):

        if self.model_type == "LEQE-CNN" or self.model_type == "L-CNN" or self.model_type == "E-CNN":
            print("image rot", tensor.shape)

            # Handle (1, H, W, C) by squeezing the batch dimension
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)  # (H, W, C)

            tensor = tensor.permute(2, 0, 1)  # (C, H, W)
            transformed = TF.rotate(tensor, angle)

            if flip == "h":
                transformed = TF.hflip(transformed)
            elif flip == "v":
                transformed = TF.vflip(transformed)

            # Ensure the result has the shape (H, W, C)
            transformed = transformed.permute(1, 2, 0)  # (H, W, C)

            # Add the batch dimension back to (1, H, W, C)
            return transformed.unsqueeze(0)  # (1, H, W, C)

        elif self.model_type == "EQE-CNN" and tensor.dim() <= 4:
            transformed = TF.rotate(tensor, angle)
            if flip == "h":
                transformed = TF.hflip(transformed)
            elif flip == "v":
                transformed = TF.vflip(transformed)
            return transformed

        elif self.model_type == "EQE-CNN" and tensor.dim() == 5:
            B, C, G, H, W = tensor.shape
            transformed = []
            for g in range(G):
                x_g = tensor[:, :, g, :, :]
                x_g = TF.rotate(x_g, angle)
                if flip == "h":
                    x_g = TF.hflip(x_g)
                elif flip == "v":
                    x_g = TF.vflip(x_g)
                transformed.append(x_g.unsqueeze(2))
            return torch.cat(transformed, dim=2)

        elif self.model_type == "LEQE-CNN-2" and tensor.dim() <= 4:

            # Handle (1, H, W, C) by squeezing the batch dimension
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)  # (H, W, C)

            tensor = tensor.permute(2, 0, 1)  # (C, H, W)
            transformed = TF.rotate(tensor, angle)

            if flip == "h":
                transformed = TF.hflip(transformed)
            elif flip == "v":
                transformed = TF.vflip(transformed)

            # Ensure the result has the shape (H, W, C)
            transformed = transformed.permute(1, 2, 0)  # (H, W, C)

            # Add the batch dimension back to (1, H, W, C)
            return transformed.unsqueeze(0)  # (1, H, W, C)

        elif self.model_type == "LEQE-CNN-2" and tensor.dim() == 5:
            B, G, H, W, C= tensor.shape
            tensor = tensor.permute(0,4,1,2,3)
            transformed = []
            for g in range(G):
                x_g = tensor[:, :, g, :, :]
                x_g = TF.rotate(x_g, angle)
                if flip == "h":
                    x_g = TF.hflip(x_g)
                elif flip == "v":
                    x_g = TF.vflip(x_g)

                transformed.append(x_g.unsqueeze(2))
            final = torch.cat(transformed, dim=2)
            final = final.permute(0,2,3,4,1)
            return final
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


    def test_equivariance_layer(self,layer_name, layer, data, visualize=True, save_path="./equivariance_vis", sample_idx=0, num_channels=3, logger=None,model_type="",eq_type="P4"):
        
        print("type: ", self.model_type)
        layer.eval()
        data = data.to(self.device)
        if model_type == "LEQE-CNN" or model_type == "L-CNN" or model_type == "E-CNN":
            _, h, w, c = data.shape
        elif model_type == "EQE-CNN":
            if layer_name == "module.encoder.conv1":
                _, c, h, w = data.shape
            else:
                _, c, g, h, w = data.shape
        elif model_type == "LEQE-CNN-2":
            if layer_name == "module.encoder.conv1":
                _, h, w, c = data.shape
            else:
                _, g, h, w, c = data.shape

        if visualize:
            if c <= 3 and data.ndim <= 4:
                plot_image(data[0], save_path=save_path,name="original",model_type=model_type)
            else:
                visalize_group_feature(data[0], save_path=save_path, name ="original",model_type=model_type)


        if eq_type == "P4":
            rotations = [0, 90, 180, 270]
            reflections = []
        else:
            rotations = [0, 90, 180, 270]
            reflections = ['h', 'v']

        all_outputs = []
        transformed_outputs = []
        transform_names = []
        errors = []

        with torch.no_grad():
            original_out = layer(data)
            if visualize:
                print("shape",original_out.shape)
                visalize_group_feature(original_out[0], save_path=save_path, name ="original_layer",model_type=model_type)

            if not isinstance(original_out, torch.Tensor):
                original_out = getattr(original_out, "tensor", original_out)

        for rot in rotations:
            x_rot = self.apply_transformation(data.clone(), angle=rot)
            if visualize and rot == 90:
                if c > 3:
                    visalize_group_feature( x_rot[0], save_path=save_path, name ="original_rot",model_type=model_type)
                else:
                    print("x_rot", x_rot.shape)

                    plot_image( x_rot[0], save_path=save_path,name="original_rot",model_type=model_type)


            with torch.no_grad():
                out_trans = layer(x_rot)
                out_trans = getattr(out_trans, "tensor", out_trans)
                if visualize and rot == 90:
                    visalize_group_feature(out_trans[0], save_path=save_path, name ="rot_layer",model_type=model_type)

            expected = self.apply_transformation(original_out.clone(), angle=rot)
            if visualize and rot == 90:
                visalize_group_feature(expected[0], save_path=save_path, name ="layer_rot",model_type=model_type)


            all_outputs.append(expected[0])
            transformed_outputs.append(out_trans[0])
            errors.append(F.mse_loss(expected, out_trans).item())
            transform_names.append(f"rot_{rot}deg")

        for refl in reflections:
            x_ref = self.apply_transformation(data.clone(), flip=refl)
            with torch.no_grad():
                out_trans = layer(x_ref)
                out_trans = getattr(out_trans, "tensor", out_trans)

            expected = self.apply_transformation(original_out.clone(), flip=refl)

            all_outputs.append(expected[0])
            transformed_outputs.append(out_trans[0])
            errors.append(F.mse_loss(expected, out_trans).item())
            transform_names.append(f"flip_{refl}")

        if visualize:
            layer_name = layer.__class__.__name__
            layer_folder = os.path.join(save_path, layer_name)
            os.makedirs(layer_folder, exist_ok=True)
            visualize_equivariance(all_outputs, transformed_outputs, rotations, reflections, layer_folder,model_type=model_type)

        mean_error = sum(errors) / len(errors)

        log = logger.log_entry if logger else print
        log("Equivariance Errors:")
        for name, err in zip(transform_names, errors):
            log(f"  {name}: {err:.6f}")
        log(f"Mean Equivariance Error: {mean_error:.6f}")

        return mean_error


    def test_model(self, dataloader):
        self.model.eval()
        data, _ = next(iter(dataloader))
        image = data[0].unsqueeze(0).to(self.device)

        inputs_by_layer = {}

        def save_input_hook(name):
            def hook(module, input, output):
                inputs_by_layer[name] = input[0].detach()
            return hook

        hooks = []
        for name, layer in self.model.named_modules():
            
            if isinstance(layer, self.layer_types):
                
                hooks.append(layer.register_forward_hook(save_input_hook(name)))
        
        with torch.no_grad():
            self.model(image)

        results = []
        for name, layer in self.model.named_modules():
            
            if not isinstance(layer, self.layer_types):
                continue

            input_tensor = inputs_by_layer.get(name)
            if input_tensor is None:
                self.logger.log_entry(f"[!] Skipped layer {name} (no input captured)")
                continue

            layer_path = os.path.join(self.save_path, name)
            os.makedirs(layer_path, exist_ok=True)
            self.logger.log_entry(f"Testing layer: {name}")

            error = self.test_equivariance_layer(name, layer, input_tensor, visualize=True, save_path=layer_path,logger= self.logger, model_type=self.model_type, eq_type=self.eq_type)
            self.logger.log_entry(f"  -> Error: {error:.6f}")
            results.append((name, error))

        for hook in hooks:
            hook.remove()

        self.logger.save()
        return results


# Example usage:
# tester = EquivarianceTester(model, device, model_type="LEQE-CNN")
# tester.test_model(dataloader)
