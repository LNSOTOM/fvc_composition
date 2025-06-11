import os
import torch
from tqdm import tqdm
import numpy as np
import rasterio

from dataset.calperum_dataset import CalperumDataset
from dataset.data_augmentation_wrapper import AlbumentationsTorchWrapper
from dataset.data_augmentation import get_train_augmentation, save_augmented_pair
import config_param

#rgb
# def debug_visualize(image_np, mask_np, idx):
#     """Create visualization of the augmented image for debugging"""
#     import matplotlib.pyplot as plt
    
#     # For multispectral data, create a RGB composite using bands 3,2,1 (if they exist)
#     rgb = None
#     if image_np.shape[0] >= 3:
#         # Use first 3 bands for visualization, scale to 0-255
#         rgb = image_np[:3].transpose(1, 2, 0).copy()
#         for i in range(3):
#             band = rgb[:,:,i]
#             p2 = np.percentile(band, 2)  # Dark end percentile
#             p98 = np.percentile(band, 98)  # Bright end percentile
#             rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     if rgb is not None:
#         plt.imshow(rgb)
#     else:
#         # Use first band if RGB composite isn't possible
#         plt.imshow(image_np[0], cmap='viridis')
#     plt.title("Augmented Image")
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask_np, cmap='tab20')
#     plt.title("Augmented Mask")
    
#     plt.savefig(f"debug_aug_{idx}.png")
#     plt.close()

#multispectral without orginal
# def debug_visualize(image_np, mask_np, idx):
#     """Create visualization of the augmented image for debugging"""
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import ListedColormap
    
#     # Define the dense site color palette
#     class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # Colors for BE, NPV, PV, SI, WI classes (dense)
#     cmap = ListedColormap(class_colors)
    
#     # For multispectral data, create a RGB composite using bands 5,3,1 (if they exist)
#     rgb = None
#     if image_np.shape[0] >= 5:  # Make sure we have at least 5 bands
#         # Use bands 5,3,1 for visualization (0-indexed: bands 4,2,0)
#         band_indices = [4, 2, 0]  # 5th, 3rd, 1st bands (0-indexed)
#         rgb = np.zeros((image_np.shape[1], image_np.shape[2], 3))
        
#         for i, band_idx in enumerate(band_indices):
#             band = image_np[band_idx].copy()
#             p2 = np.percentile(band, 2)  # Dark end percentile
#             p98 = np.percentile(band, 98)  # Bright end percentile
#             rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
#     elif image_np.shape[0] >= 3:
#         # Fallback to first 3 bands if we don't have 5 bands
#         rgb = image_np[:3].transpose(1, 2, 0).copy()
#         for i in range(3):
#             band = rgb[:,:,i]
#             p2 = np.percentile(band, 2)
#             p98 = np.percentile(band, 98)
#             rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
#     plt.figure(figsize=(15, 5))  # Made wider to accommodate colorbar
#     plt.subplot(1, 2, 1)
#     if rgb is not None:
#         plt.imshow(rgb)
#         plt.title("Augmented Image (Bands 5,3,1)")
#     else:
#         # Use first band if RGB composite isn't possible
#         plt.imshow(image_np[0], cmap='viridis')
#         plt.title("Augmented Image (Band 1)")
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     im = plt.imshow(mask_np, cmap=cmap, interpolation='nearest')
#     plt.title("Augmented Mask")
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     im = plt.imshow(mask_np, cmap=cmap, interpolation='nearest')
#     plt.title("Augmented Mask")
#     plt.axis('off')
    
#     # Add colorbar with proper labels
#     colorbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
#     colorbar.set_ticks([0, 1, 2, 3, 4])  # Set ticks for 5 classes (0-4)
#     colorbar.ax.set_yticklabels(['BE class', 'NPV class', 'PV class', 'SI class', 'WI class'], fontsize=10)
    
#     plt.tight_layout()
#     plt.savefig(f"debug_aug_{idx}.png", dpi=150, bbox_inches='tight')
#     plt.close()

#Multispectral with original
def debug_visualize(orig_image_np, aug_image_np, aug_mask_np, idx):
    """Create visualization comparing original and augmented image with mask"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Define the dense site color palette
    class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # Colors for BE, NPV, PV, SI, WI classes (dense)
    cmap = ListedColormap(class_colors)
    
    # Function to create RGB composite
    def create_rgb_composite(image_np):
        rgb = None
        if image_np.shape[0] >= 5:  # Make sure we have at least 5 bands
            # Use bands 5,3,1 for visualization (0-indexed: bands 4,2,0)
            band_indices = [4, 2, 0]  # 5th, 3rd, 1st bands (0-indexed)
            rgb = np.zeros((image_np.shape[1], image_np.shape[2], 3))
            
            for i, band_idx in enumerate(band_indices):
                band = image_np[band_idx].copy()
                p2 = np.percentile(band, 2)  # Dark end percentile
                p98 = np.percentile(band, 98)  # Bright end percentile
                rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        elif image_np.shape[0] >= 3:
            # Fallback to first 3 bands if we don't have 5 bands
            rgb = image_np[:3].transpose(1, 2, 0).copy()
            for i in range(3):
                band = rgb[:,:,i]
                p2 = np.percentile(band, 2)
                p98 = np.percentile(band, 98)
                rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        return rgb
    
    # Create RGB composites
    orig_rgb = create_rgb_composite(orig_image_np)
    aug_rgb = create_rgb_composite(aug_image_np)
    
    plt.figure(figsize=(20, 6))  # Made wider to accommodate 3 subplots
    
    # Original image
    plt.subplot(1, 3, 1)
    if orig_rgb is not None:
        plt.imshow(orig_rgb)
        plt.title("Original Image (Bands 5,3,1)")
    else:
        plt.imshow(orig_image_np[0], cmap='viridis')
        plt.title("Original Image (Band 1)")
    plt.axis('off')
    
    # Augmented image
    plt.subplot(1, 3, 2)
    if aug_rgb is not None:
        plt.imshow(aug_rgb)
        plt.title("Augmented Image (Bands 5,3,1)")
    else:
        plt.imshow(aug_image_np[0], cmap='viridis')
        plt.title("Augmented Image (Band 1)")
    plt.axis('off')
    
    # Augmented mask
    plt.subplot(1, 3, 3)
    im = plt.imshow(aug_mask_np, cmap=cmap, interpolation='nearest')
    plt.title("Augmented Mask")
    plt.axis('off')
    
    # Add colorbar with proper labels
    colorbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
    colorbar.set_ticks([0, 1, 2, 3, 4])  # Set ticks for 5 classes (0-4)
    colorbar.ax.set_yticklabels(['BE class', 'NPV class', 'PV class', 'SI class', 'WI class'], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"debug_aug_{idx}.png", dpi=150, bbox_inches='tight')
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
            # debug_visualize(aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
            # Debug visualization with original image included
            debug_visualize(image_np, aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
            
            # Save
            save_augmented_pair(orig_img_path, orig_mask_path, aug_image_np, aug_mask_np, aug_idx, output_dir_img, output_dir_mask)

if __name__ == "__main__":
    main()