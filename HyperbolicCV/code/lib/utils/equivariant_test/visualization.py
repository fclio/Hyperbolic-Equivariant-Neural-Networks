import os
import matplotlib.pyplot as plt
import math
import torch
import io
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

def plot_image(image, save_path, name="test", model_type=""):
    """
    Plots and saves image or all feature map channels.
    image: Tensor [C, H, W] or [H, W]
    """
    os.makedirs(save_path, exist_ok=True)
    # Check if image is a tensor (PyTorch) or numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()  # Move to CPU if it's a tensor
    elif isinstance(image, np.ndarray):
        # If image is already a numpy array, no need to move it to CPU
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Must be a Tensor or numpy array.")


    # Single grayscale image [1, H, W] -> [H, W]
    if image.shape[0] == 1:
        print("plot image 1", image.shape)
        image = image[0]

        # Handle model-specific reshaping
    if (model_type == "LEQE-CNN" or model_type == "LEQE-CNN-2" or model_type == "L-CNN" or model_type == "E-CNN") and image.ndim == 3:
        image = image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

    # Case: Multi-channel feature map [C, H, W]
    if image.ndim == 3:
        print("plot image 2", image.shape)
        C, H, W = image.shape
        cols = math.ceil(math.sqrt(C))
        rows = math.ceil(C / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

        for i in range(rows * cols):
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
            if i < C:
                ax.imshow(image[i], cmap='viridis')
                ax.set_title(f"{name} - Ch {i}")
            ax.axis('off')

        plt.tight_layout()
        file_path = os.path.join(save_path, f"{name}_all_channels.png")
        plt.savefig(file_path)
        plt.close()

    # Case: Single 2D image [H, W]
    elif image.ndim == 2:
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='viridis')
        plt.title(name)
        plt.axis('off')
        file_path = os.path.join(save_path, f"{name}.png")
        plt.savefig(file_path)
        plt.close()

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

def visualize_feature_maps_to_image(tensor, cols=8, cmap='viridis'):

    """
    Converts a (C, H, W) tensor into a PIL image showing a grid of feature maps.
    """
    C, H, W = tensor.shape
    rows = (C + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis('off')
        if i < C:
            img = tensor[i].detach().cpu()
            axes[i].imshow(img, cmap=cmap)

    plt.tight_layout()
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return pil_img

def visalize_group_feature(x, save_path, name="", model_type=""):
    if model_type == "LEQE-CNN":
        visalize_group_LEQE(x,save_path,name,grid_rows=3, grid_cols=3,model_type=model_type)
    elif model_type =="EQE-CNN":
        visalize_group_EQE(x,save_path,name,grid_rows=2, grid_cols=4,model_type=model_type)
    elif model_type == "LEQE-CNN-2":
        x = x.permute(3, 0, 1, 2)  # (C, 8, H, W)
        visalize_group_EQE(x,save_path,name,grid_rows=1, grid_cols=4,model_type=model_type)
    elif model_type == "L-CNN" or  model_type == "E-CNN" :
        visalize_group_single(x,save_path, name)

def visalize_group_LEQE(x, save_path, name="", grid_rows=2, grid_cols=4, model_type=""):
    h, w, c = x.shape
    spatial_channel = int((c - 1) / 8)

    # Step 1: Extract the time component (H, W)
    time_component = x[:, :, -1].detach().cpu().numpy()

    # Step 2: Extract spatial component and reshape
    spatial_flat = x[:, :, :-1]
    spatial_reshaped = spatial_flat.view(h, w, 8, spatial_channel)  # (H, W, 8, C)
    spatial_reshaped = spatial_reshaped.permute(3, 2, 0, 1)  # (8, C, H, W)

    # Pass time_component as optional image
    visalize_group_EQE(spatial_reshaped, save_path, name=name, grid_rows=grid_rows, grid_cols=grid_cols, model_type=model_type, time_image=time_component)

def visalize_group_EQE(x, save_path, name="", grid_rows=3, grid_cols=3, model_type="", time_image=None,eq_type="P4"):
    """
    Combine 8 group spatial maps (+ optional time image) into one big image with labels.
    """
    if eq_type =="P4M":
        labels = [
            "0°", "90°", "180°", "270°",
            "reflection", "reflection+90°", "reflection+180°", "reflection+270°"
        ]
    else:
        labels = [
        "0°", "90°", "180°", "270°"
    ]


    imgs = []
    font_size = 100
    label_height = font_size + 20

    # --- Add spatial group maps ---
    x_reordered = x.permute(1, 0, 2, 3)  # (8, C, H, W)

    for i in range(len(labels)):
        img = visualize_feature_maps_to_image(x_reordered[i])
        labeled = Image.new("RGB", (img.width, img.height + label_height), "white")
        labeled.paste(img, (0, label_height))
        draw = ImageDraw.Draw(labeled)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), labels[i], fill="black", font=font)
        imgs.append(labeled)

    # --- Add time image if provided (added last) ---
    if time_image is not None:
        labels.append("Time")
        if isinstance(time_image, torch.Tensor):
            time_image = time_image.detach().cpu().squeeze().numpy()
        else:
            time_image = time_image.squeeze()

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.axis("off")
        ax.imshow(time_image, cmap="viridis")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        time_pil = Image.open(buf).convert("RGB")
        plt.close(fig)

        # Resize to match feature map size
        time_pil = time_pil.resize(imgs[0].size)

        labeled = Image.new("RGB", (time_pil.width, time_pil.height + label_height), "white")
        labeled.paste(time_pil, (0, label_height))
        draw = ImageDraw.Draw(labeled)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), labels[-1], fill="black", font=font)
        imgs.append(labeled)

    # --- Grid layout ---
    img_w, img_h = imgs[0].size
    num_imgs = len(imgs)
    # grid_cols = math.ceil(math.sqrt(num_imgs))
    # grid_rows = math.ceil(num_imgs / grid_cols)
    grid_cols = len(labels)
    grid_rows = math.ceil(num_imgs / grid_cols)

    grid_img = Image.new("RGB", (grid_cols * img_w, grid_rows * img_h), color='white')
    for idx, img in enumerate(imgs):
        row = idx // grid_cols
        col = idx % grid_cols
        grid_img.paste(img, (col * img_w, row * img_h))

    os.makedirs(save_path, exist_ok=True)
    grid_img.save(os.path.join(save_path, f"feature_maps_{name}.png"))


def visalize_group_single(feature_map, save_path, name):
    """
    Saves a grid of feature maps (PyTorch tensor) to an image using PIL.
    feature_map: (C, H, W) or (H, W, C) tensor
    """
    if feature_map.dim() == 3 and feature_map.shape[0] != 1 and feature_map.shape[0] != 3:
        # Ensure it's (H, W, C)
        if feature_map.shape[0] < 10:  # Likely (C, H, W)
            feature_map = feature_map.permute(1, 2, 0)  # (H, W, C)

    feature_map = feature_map.detach().cpu()

    h, w, c = feature_map.shape
    cols = 9
    rows = int((c + cols - 1) / cols)

    normalized_maps = []
    for i in range(c):
        ch = feature_map[:, :, i]
        ch_min = ch.min()
        ch_max = ch.max()
        ch_norm = ((ch - ch_min) / (ch_max - ch_min + 1e-5) * 255).byte()
        normalized_maps.append(ch_norm.numpy())

    # Create empty canvas
    grid = Image.new('L', (cols * w, rows * h))

    for idx, img_array in enumerate(normalized_maps):
        img = Image.fromarray(img_array, mode='L')
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(img, (x, y))

    os.makedirs(save_path, exist_ok=True)
    grid.save(os.path.join(save_path, f"feature_maps_{name}.png"))

def visualize_single_feature_maps(tensor, name ="image", cols=8, cmap='viridis', save_path=None):
    """
    Visualizes a (C, H, W) tensor as a grid of feature maps.

    Args:
        tensor: torch.Tensor of shape (C, H, W)
        cols: number of columns in the grid
        cmap: color map for imshow
        save_path: optional path to save the image
    """
    assert tensor.dim() == 3, f"Expected shape (C, H, W), got {tensor.shape}"
    C, H, W = tensor.shape
    rows = (C + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis('off')
        if i < C:
            img = tensor[i].detach().cpu()
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(f'Ch {i}', fontsize=8)

    plt.tight_layout()
    if save_path:
        file_path = os.path.join(save_path, f"{name}.png")
        plt.savefig(file_path, bbox_inches='tight')
    plt.show()


def visualize_equivariance(original_outputs, transformed_outputs, transformations, reflections,  save_path, channel_idx=0, num_transformation=4,model_type=""):
    """
    Visualizes multiple channels of original and transformed outputs for different transformations.
    """

    total_transforms = len(transformations) + len(reflections)
    fig, axs = plt.subplots(num_transformation, 2 * total_transforms, figsize=(3 * 2 * total_transforms, 3 * num_transformation))

    for i in range(total_transforms):
        for ch in range(num_transformation):

            if model_type == "LEQE-CNN" or model_type == "L-CNN" or model_type == "E-CNN":
                orig = original_outputs[i][..., channel_idx+1].detach().cpu().numpy()  # shape (H, W)
                trans = transformed_outputs[i][..., channel_idx+1].detach().cpu().numpy()  # shape (H, W)
            elif model_type == "EQE-CNN":
                orig = original_outputs[i][channel_idx, ch].detach().cpu().numpy()
                trans = transformed_outputs[i][channel_idx, ch].detach().cpu().numpy()
            elif model_type == "LEQE-CNN-2":
                # original_outputs = original_outputs.permute(0,4,1,2,3)
                # transformed_outputs = transformed_outputs.permute(0,4,1,2,3)
                orig = original_outputs[i].permute(3,0,1,2)
                orig = orig[channel_idx, ch].detach().cpu().numpy()
                trans = transformed_outputs[i].permute(3,0,1,2)
                trans = trans[channel_idx, ch].detach().cpu().numpy()

            ax1 = axs[ch, 2 * i] if num_transformation > 1 else axs[2 * i]
            ax2 = axs[ch, 2 * i + 1] if num_transformation > 1 else axs[2 * i + 1]

            ax1.imshow(orig, cmap="viridis")
            ax2.imshow(trans, cmap="viridis")

            if ch == 0:
                title = f"Rot {transformations[i]}°" if i < len(transformations) else f"Refl {reflections[i - len(transformations)]}"
                ax1.set_title(f"Orig {title}")
                ax2.set_title(f"Transf {title}")

            ax1.axis("off")
            ax2.axis("off")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"equivariance_feature_{channel_idx}_compare.png")
    plt.savefig(file_path)
    plt.close()



