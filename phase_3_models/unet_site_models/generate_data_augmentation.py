import os
import torch
from tqdm import tqdm
import numpy as np
import rasterio

from dataset.calperum_dataset import CalperumDataset
from dataset.data_augmentation_wrapper import AlbumentationsTorchWrapper
from dataset.data_augmentation import get_train_augmentation, save_augmented_pair
from dataset.mask_preprocessing import prep_mask_preserve_nan
from dataset.image_preprocessing import load_raw_multispectral_image
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
def debug_visualize(orig_image_np, aug_image_np, aug_mask_np, idx, visualize="both"):
    """
    Visualize original and/or augmented images and masks with NaNs preserved.
    
    visualize options:
    - "both": show original, augmented image, and mask
    - "image": show only images
    - "mask": show only the mask
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    # ✅ Convert -1 to NaN
    aug_mask_np = aug_mask_np.astype(np.float32)
    aug_mask_np[aug_mask_np == -1] = np.nan

    def create_rgb(image_np):
        if image_np.shape[0] >= 5:
            indices = [4, 2, 0]
        elif image_np.shape[0] >= 3:
            indices = [2, 1, 0]
        else:
            return None
        rgb = np.zeros((image_np.shape[1], image_np.shape[2], 3))
        for i, b in enumerate(indices):
            band = image_np[b]
            p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
            rgb[..., i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        return rgb

    orig_rgb = create_rgb(orig_image_np)
    aug_rgb = create_rgb(aug_image_np)

    # ---------------- Print mask stats including NaN ----------------
    nan_count = np.isnan(aug_mask_np).sum()
    unique_vals = np.unique(aug_mask_np[~np.isnan(aug_mask_np)])
    nan_first_list = (['NaN'] if nan_count > 0 else []) + list(unique_vals)

    print(f"\n=== DEBUG {idx} MASK ANALYSIS ===")
    print(f"Mask dtype: {aug_mask_np.dtype}")
    print(f"Mask shape: {aug_mask_np.shape}")
    print(f"Mask min: {np.nanmin(aug_mask_np):.4f}, max: {np.nanmax(aug_mask_np):.4f}")
    print(f"NaN count: {nan_count}")
    print(f"Unique values (excluding NaN): {unique_vals}")
    print(f"Unique values (with NaN): {nan_first_list}")

    # ---------------- Visualization Mode Setup ----------------
    class_labels = ['BE', 'NPV', 'PV', 'SI', 'WI']
    class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']
    cmap = ListedColormap(class_colors[:len(class_labels)])

    valid_mask = ~np.isnan(aug_mask_np)
    masked_mask = np.ma.masked_array(
        aug_mask_np,
        mask=np.isnan(aug_mask_np)  # Mask where values are NaN
    )

    if visualize == "both":
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        titles = ["Original Image", "Augmented Image", "Augmented Mask"]
        panels = [orig_rgb, aug_rgb, masked_mask]
        cmaps = [None, None, cmap]

    elif visualize == "image":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        titles = ["Original Image", "Augmented Image"]
        panels = [orig_rgb, aug_rgb]
        cmaps = [None, None]

    elif visualize == "mask":
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes = [axes]  # ensure iterable
        titles = ["Augmented Mask"]
        panels = [masked_mask]
        cmaps = [cmap]

    else:
        raise ValueError(f"Invalid visualize option: {visualize}")

    # ---------------- Plot panels ----------------
    for ax, panel, title, cm in zip(axes, panels, titles, cmaps):
        if title == "Augmented Mask":
            # First plot white background
            ax.imshow(np.ones_like(aug_mask_np), cmap='gray', vmin=0, vmax=1)
            
            # Then plot only the valid values with correct color mapping
            im = ax.imshow(masked_mask, cmap=cm, interpolation='nearest', vmin=0, vmax=len(class_labels)-1)
            
            # Add hatching to NaN areas for better visibility
            if np.any(np.isnan(aug_mask_np)):
                ax.contourf(np.isnan(aug_mask_np), hatches=['//'], colors='none', alpha=0.15)
            
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.045, pad=0.04)
            cbar.set_ticks(np.arange(len(class_labels)))
            cbar.ax.set_yticklabels(class_labels)
        else:
            if panel is not None:
                ax.imshow(panel)
            else:
                ax.imshow(orig_image_np[0], cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"debug_aug_{idx}.png", dpi=150, bbox_inches='tight')
    plt.close()






# def main():
#     # Make sure output directories exist
#     output_dir_img= '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/predictor_5b'
#     output_dir_mask= '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
#     os.makedirs(output_dir_img, exist_ok=True)
#     os.makedirs(output_dir_mask, exist_ok=True)

#     image_dirs = config_param.IMAGE_FOLDER
#     mask_dirs = config_param.MASK_FOLDER
#     # 1. Create dataset for originals (no transforms)
#     dataset = CalperumDataset(
#         image_folders=image_dirs,
#         mask_folders=mask_dirs,
#         transform=None
#     )

#     # 2. Create Albumentations transform
#     # a) Create augmentation wrapper - it will automatically use get_train_augmentation()
#     albumentations_wrapper = AlbumentationsTorchWrapper()
    
#     # b) Alternative: explicitly pass the transform if you want to be clear
#     # from dataset.data_augmentation import get_train_augmentation
#     # albumentations_transform = get_train_augmentation()
#     # albumentations_wrapper = AlbumentationsTorchWrapper(albumentations_transform)

#     # 3. Decide how many augmentations per sample
#     NUM_AUG_PER_IMAGE = 2

#     # 4. Iterate and generate augmentations
#     print("Generating and saving augmented data...")
#     orig_img_dir = image_dirs[0]
#     orig_mask_dir = mask_dirs[0]
#     orig_img_files = sorted([f for f in os.listdir(orig_img_dir) if f.endswith('.tif')])
#     orig_mask_files = sorted([f for f in os.listdir(orig_mask_dir) if f.endswith('.tif')])

#     for idx in tqdm(range(len(dataset)), desc="Processing images"):
#         image_tensor, mask_tensor = dataset[idx]
#           # Debug: Check loaded tensor values
#         print(f"Loaded tensor {idx}: min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}")
        
#         # Convert to numpy for rasterio saving
#         # [C,H,W] -> rasterio expects [C,H,W] for multiband, mask [H,W]
#         image_np = image_tensor.numpy()
#         print(f"Original image {idx}: min={image_np.min():.4f}, max={image_np.max():.4f}, mean={image_np.mean():.4f}")
#         mask_np = mask_tensor.numpy()

#         orig_img_path = os.path.join(orig_img_dir, orig_img_files[idx])
#         orig_mask_path = os.path.join(orig_mask_dir, orig_mask_files[idx])

#         for aug_idx in tqdm(range(1, NUM_AUG_PER_IMAGE+1), 
#                     desc=f"Augmenting image {idx}", 
#                     leave=False):
#             # Apply augmentation
#             (aug_image_tensor, aug_mask_tensor), (aug_image_np, aug_mask_np) = albumentations_wrapper(image_tensor, mask_tensor)
    
#             # aug_image, aug_mask = albumentations_wrapper(image_tensor, mask_tensor)
#             # aug_image_np = aug_image.numpy()
#             # aug_mask_np = aug_mask.numpy()
            
#             # Print stats post-augmentation
#             print(f"  Aug {aug_idx}: min={aug_image_np.min():.4f}, max={aug_image_np.max():.4f}, mean={aug_image_np.mean():.4f}")
                   
#             # Check for all-zero channels
#             for c in range(aug_image_np.shape[0]):
#                 channel = aug_image_np[c]
#                 if channel.min() == 0 and channel.max() == 0:
#                     print(f"  WARNING: Channel {c} is all zeros!")
                    
#             # Debug visualization
#             # debug_visualize(aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
#             # Debug visualization with original image included
#             debug_visualize(image_np, aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
            
#             # Save
#             save_augmented_pair(orig_img_path, orig_mask_path, aug_image_np, aug_mask_np, aug_idx, output_dir_img, output_dir_mask)

###test 5 files
# def main():
#     # Make sure output directories exist
#     output_dir_img= '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/predictor_5b'
#     output_dir_mask= '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
#     os.makedirs(output_dir_img, exist_ok=True)
#     os.makedirs(output_dir_mask, exist_ok=True)

#     image_dirs = config_param.IMAGE_FOLDER
#     mask_dirs = config_param.MASK_FOLDER
#     # 1. Create dataset for originals (no transforms)
#     dataset = CalperumDataset(
#         image_folders=image_dirs,
#         mask_folders=mask_dirs,
#         transform=None
#     )

#     # 2. Create Albumentations transform
#     # a) Create augmentation wrapper - it will automatically use get_train_augmentation()
#     albumentations_transform = get_train_augmentation() 
#     albumentations_wrapper = AlbumentationsTorchWrapper(albumentations_transform)


#     # 3. Decide how many augmentations per sample
#     NUM_AUG_PER_IMAGE = 2
    
#     # 🎯 TESTING: Process only first 5 images
#     NUM_TEST_IMAGES = 5
#     total_images = min(NUM_TEST_IMAGES, len(dataset))
    
#     print(f"🧪 TESTING MODE: Processing only {total_images} images out of {len(dataset)} total")

#     # 4. Iterate and generate augmentations
#     print("Generating and saving augmented data...")
#     orig_img_dir = image_dirs[0]
#     orig_mask_dir = mask_dirs[0]
#     orig_img_files = sorted([f for f in os.listdir(orig_img_dir) if f.endswith('.tif')])
#     orig_mask_files = sorted([f for f in os.listdir(orig_mask_dir) if f.endswith('.tif')])

#     # 🎯 MODIFIED: Only process first 5 images
#     # In your generate_data_augmentation.py main() function:

#     # Then in the main() function:
#     for idx in tqdm(range(total_images), desc="Processing images"):
#         # 1. Load image and mask directly as NumPy arrays (not tensors)
#         img_filename = os.path.join(orig_img_dir, orig_img_files[idx])
#         mask_filename = os.path.join(orig_mask_dir, orig_mask_files[idx])
        
#         # Load directly as NumPy arrays with NaN preserved
#         image_np, _ = load_raw_multispectral_image(img_filename)
#         mask_np, _ = prep_mask_preserve_nan(mask_filename)  # Use preserve_nan version!
        
#         print(f"Original image {idx}: min={image_np.min():.4f}, max={image_np.max():.4f}, mean={image_np.mean():.4f}")
#         print(f"Original mask {idx}: shape={mask_np.shape}, dtype={mask_np.dtype}")
#         print(f"  - NaN values: {np.sum(np.isnan(mask_np))}")
#         print(f"  - Unique values: {np.unique(mask_np[~np.isnan(mask_np)])}")
        
#         for aug_idx in tqdm(range(1, NUM_AUG_PER_IMAGE+1), 
#                     desc=f"Augmenting image {idx}", 
#                     leave=False):
            
#             # 2. Apply augmentation directly with NumPy arrays
#             aug_image_np, aug_mask_np = albumentations_wrapper(image_np, mask_np)
            
#             # 3. Debug visualization with NumPy arrays
#             debug_visualize(image_np, aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
            
#             # 4. Save with NumPy arrays (NaN preserved)
#             save_augmented_pair(img_filename, mask_filename, aug_image_np, aug_mask_np, aug_idx, output_dir_img, output_dir_mask)
        
#         # 📊 Summary
#         print(f"\n✅ TESTING COMPLETE!")
#         print(f"   - Processed: {total_images} original images")
#         print(f"   - Generated: {total_images * NUM_AUG_PER_IMAGE} augmented images") 
#         print(f"   - Debug visualizations: {total_images * NUM_AUG_PER_IMAGE} PNG files")
#         print(f"   - Total files created: {total_images * NUM_AUG_PER_IMAGE * 2} (images + masks)")

def main():
    # Make sure output directories exist
    output_dir_img = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/predictor_5b'
    output_dir_mask = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)

    # image_dirs = config_param.IMAGE_FOLDER
    # mask_dirs = config_param.MASK_FOLDER
    image_dirs = config_param.SUBSAMPLE_IMAGE_DIR
    mask_dirs = config_param.SUBSAMPLE_MASK_DIR

    # 1. Create dataset for originals (no transforms)/ susample dataset
    dataset = CalperumDataset(
        image_folders=image_dirs,
        mask_folders=mask_dirs,
        transform=None
    )

    # 2. Create Albumentations transform and wrapper
    albumentations_transform = get_train_augmentation() 
    # Without debug output (for production)
    # albumentations_wrapper = AlbumentationsTorchWrapper(albumentations_transform, debug=False)
    # With debug output (for development/testing)
    albumentations_wrapper = AlbumentationsTorchWrapper(albumentations_transform, debug=True)

    
    # 3. Decide how many augmentations per sample
    NUM_AUG_PER_IMAGE = 2

    # Process all images in the dataset
    total_images = len(dataset)
    print(f"🚀 Processing all {total_images} images in the dataset")

    # 4. Iterate and generate augmentations
    print("Generating and saving augmented data...")
    orig_img_dir = image_dirs[0]
    orig_mask_dir = mask_dirs[0]
    orig_img_files = sorted([f for f in os.listdir(orig_img_dir) if f.endswith('.tif')])
    orig_mask_files = sorted([f for f in os.listdir(orig_mask_dir) if f.endswith('.tif')])

    for idx in tqdm(range(total_images), desc="Processing images"):
        # 1. Load image and mask directly as NumPy arrays (not tensors)
        img_filename = os.path.join(orig_img_dir, orig_img_files[idx])
        mask_filename = os.path.join(orig_mask_dir, orig_mask_files[idx])
        
        # Load directly as NumPy arrays with NaN preserved
        image_np, _ = load_raw_multispectral_image(img_filename)
        mask_np, _ = prep_mask_preserve_nan(mask_filename)
        
        print(f"Original image {idx}: min={image_np.min():.4f}, max={image_np.max():.4f}, mean={image_np.mean():.4f}")
        print(f"Original mask {idx}: shape={mask_np.shape}, dtype={mask_np.dtype}")
        print(f"  - NaN values: {np.sum(np.isnan(mask_np))}")
        print(f"  - Unique values: {np.unique(mask_np[~np.isnan(mask_np)])}")
        
        for aug_idx in tqdm(range(1, NUM_AUG_PER_IMAGE+1), 
                            desc=f"Augmenting image {idx}", 
                            leave=False):
            # 2. Apply augmentation directly with NumPy arrays
            aug_image_np, aug_mask_np = albumentations_wrapper(image_np, mask_np)
            
            # 3. Debug visualization with NumPy arrays
            debug_visualize(image_np, aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
            
            # 4. Save with NumPy arrays (NaN preserved)
            save_augmented_pair(img_filename, mask_filename, aug_image_np, aug_mask_np,
                                aug_idx, output_dir_img, output_dir_mask)
    
    # 📊 Summary
    print(f"\n✅ AUGMENTATION COMPLETE!")
    print(f"   - Processed: {total_images} original images")
    print(f"   - Generated: {total_images * NUM_AUG_PER_IMAGE} augmented images")
    print(f"   - Debug visualizations: {total_images * NUM_AUG_PER_IMAGE} PNG files")
    print(f"   - Total files created: {total_images * NUM_AUG_PER_IMAGE * 2} (images + masks)")



if __name__ == "__main__":
    main()