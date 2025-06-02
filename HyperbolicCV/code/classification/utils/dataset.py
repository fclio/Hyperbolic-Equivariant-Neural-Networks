import os
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Define a custom dataset class to apply transformations
class CIFAR100LT(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']
        label = item['fine_label']
        if self.transform:
            image = self.transform(image)
        return image, label


class CIFAR10LT(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label = self.dataset[actual_idx]
        if self.transform:
            img = self.transform(img)
        return img, label
import torch
import numpy as np
from PIL import Image

def save_image(img_tensor, filepath, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Saves a single PyTorch image tensor to a file.

    Args:
        img_tensor (Tensor): Image tensor of shape (3, H, W) (typically after transforms).
        filepath (str): Path to save the image, e.g., 'output/image.png'.
        mean (list): Normalization mean used during preprocessing.
        std (list): Normalization std used during preprocessing.
    """
    # Unnormalize
    img = img_tensor.clone().detach().cpu().numpy()
    img = img.transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Convert to uint8 and save
    img = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img)

    # Ensure output directory exists
    os.makedirs(filepath, exist_ok=True)

    # Create full path with filename and extension
    save_path = os.path.join(filepath, "sample.png")

    img_pil.save(save_path)
    print(f"Image saved to {filepath}")
