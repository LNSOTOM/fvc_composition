import os
import torch
from tqdm import tqdm
import numpy as np
import rasterio

from dataset.calperum_dataset import CalperumDataset
from dataset.data_augmentation_wrapper import AlbumentationsTorchWrapper
from dataset.data_augmentation import get_train_augmentation, save_augmented_pair
import config_param

def debug_visualize(image_np, mask_np, idx):
    """Create visualization of the augmented image for debugging"""
    import matplotlib.pyplot as plt
    
    # For multispectral data, create a RGB composite using bands 3,2,1 (if they exist)
    rgb = None
    if image_np.shape[0] >= 3:
        # Use first 3 bands for visualization, scale to 0-255
        rgb = image_np[:3].transpose(1, 2, 0).copy()
        for i in range(3):
            band = rgb[:,:,i]
            p2 = np.percentile(band, 2)  # Dark end percentile
            p98 = np.percentile(band, 98)  # Bright end percentile
            rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if rgb is not None:
        plt.imshow(rgb)
    else:
        # Use first band if RGB composite isn't possible
        plt.imshow(image_np[0], cmap='viridis')
    plt.title("Augmented Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='tab20')
    plt.title("Augmented Mask")
    
    plt.savefig(f"debug_aug_{idx}.png")
    plt.close()

def main():
    # Make sure output directories exist
    output_dir_img= '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/predictor_5b'
    output_dir_mask= '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)

    image_dirs = config_param.IMAGE_FOLDER
    mask_dirs = config_param.MASK_FOLDER
    # 1. Create dataset for originals (no transforms)
    dataset = CalperumDataset(
        image_folders=image_dirs,
        mask_folders=mask_dirs,
        transform=None
    )

    # 2. Create Albumentations transform
    albumentations_transform = get_train_augmentation()
    albumentations_wrapper = AlbumentationsTorchWrapper(albumentations_transform)

    # 3. Decide how many augmentations per sample
    NUM_AUG_PER_IMAGE = 2

    # 4. Iterate and generate augmentations
    print("Generating and saving augmented data...")
    orig_img_dir = image_dirs[0]
    orig_mask_dir = mask_dirs[0]
    orig_img_files = sorted([f for f in os.listdir(orig_img_dir) if f.endswith('.tif')])
    orig_mask_files = sorted([f for f in os.listdir(orig_mask_dir) if f.endswith('.tif')])

    for idx in tqdm(range(len(dataset))):
        image_tensor, mask_tensor = dataset[idx]
          # Debug: Check loaded tensor values
        print(f"Loaded tensor {idx}: min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}")
        
        # Convert to numpy for rasterio saving
        # [C,H,W] -> rasterio expects [C,H,W] for multiband, mask [H,W]
        image_np = image_tensor.numpy()
        print(f"Original image {idx}: min={image_np.min():.4f}, max={image_np.max():.4f}, mean={image_np.mean():.4f}")
        mask_np = mask_tensor.numpy()

        orig_img_path = os.path.join(orig_img_dir, orig_img_files[idx])
        orig_mask_path = os.path.join(orig_mask_dir, orig_mask_files[idx])

        for aug_idx in range(1, NUM_AUG_PER_IMAGE+1):
            # Apply augmentation
            aug_image, aug_mask = albumentations_wrapper(image_tensor, mask_tensor)
            aug_image_np = aug_image.numpy()
            aug_mask_np = aug_mask.numpy()
            
            # Print stats post-augmentation
            print(f"  Aug {aug_idx}: min={aug_image_np.min():.4f}, max={aug_image_np.max():.4f}, mean={aug_image_np.mean():.4f}")
                   
            # Check for all-zero channels
            for c in range(aug_image_np.shape[0]):
                channel = aug_image_np[c]
                if channel.min() == 0 and channel.max() == 0:
                    print(f"  WARNING: Channel {c} is all zeros!")
            # Debug visualization
            debug_visualize(aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
            # Save
            save_augmented_pair(orig_img_path, orig_mask_path, aug_image_np, aug_mask_np, aug_idx, output_dir_img, output_dir_mask)

if __name__ == "__main__":
    main()