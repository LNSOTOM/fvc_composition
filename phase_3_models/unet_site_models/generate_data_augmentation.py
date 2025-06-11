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
    import numpy as np
    
    # 🔍 DEBUG: Check what's actually in the mask
    print(f"\n=== DEBUG {idx} MASK ANALYSIS ===")
    print(f"Mask shape: {aug_mask_np.shape}")
    print(f"Mask dtype: {aug_mask_np.dtype}")
    print(f"Mask min: {np.nanmin(aug_mask_np)}, max: {np.nanmax(aug_mask_np)}")
    
    # Check for different types of "no data"
    num_nan = np.sum(np.isnan(aug_mask_np))
    num_neg_one = np.sum(aug_mask_np == -1)
    num_zero = np.sum(aug_mask_np == 0)
    
    print(f"NaN values: {num_nan}")
    print(f"-1 values (converted from NaN): {num_neg_one}")
    print(f"0 values (BE class): {num_zero}")
    print(f"Unique values: {np.unique(aug_mask_np)}")
    
    # 🔍 CRITICAL CHECK: Check -1 pixels in augmented image (not black 0 pixels)
    aug_mean = np.mean(aug_image_np, axis=0)  # Average across bands
    no_data_pixels = aug_mean == -1  # Pixels with -1 value (no-data)
    
    if np.any(no_data_pixels):
        mask_values_at_no_data = aug_mask_np[no_data_pixels]
        unique_at_no_data = np.unique(mask_values_at_no_data)
        
        print(f"🔍 CHECKING: No-data pixels in augmented image (value = -1)")
        print(f"   Number of no-data pixels: {np.sum(no_data_pixels)}")
        print(f"   Mask values at no-data pixels: {unique_at_no_data}")
        
        # 🎯 CHECK: Are no-data areas properly marked as -1 in mask?
        if -1 in unique_at_no_data and len(unique_at_no_data) == 1:
            print("✅ PERFECT: All no-data pixels have -1 (NaN) values in mask!")
        elif len(unique_at_no_data) == 0:
            print("✅ PERFECT: All no-data pixels have NaN values in mask!")
        else:
            print(f"❌ PROBLEM: No-data pixels have mixed values: {unique_at_no_data}")
    else:
        print("ℹ️  No no-data pixels (-1) detected in augmented image")
    
    # 🔍 ALSO CHECK: Make sure no fake BE class (0) pixels in borders
    black_pixels = aug_mean < 0.05  # Very dark pixels  
    if np.any(black_pixels):
        print(f"⚠️  WARNING: Found {np.sum(black_pixels)} black pixels (< 0.05)")
        print("   These might be fake BE class pixels if value=0 was used!")
    
    # 🎯 ANALYZE: Original vs Augmented areas
    orig_mean = np.mean(orig_image_np, axis=0)
    orig_black = orig_mean < 0.05
    
    # New black areas = black in augmented but not black in original
    new_black_areas = black_pixels & ~orig_black
    
    if np.any(new_black_areas):
        print(f"\n🎯 NEW BLACK AREAS FROM AUGMENTATION:")
        print(f"   Number of new black pixels: {np.sum(new_black_areas)}")
        mask_values_at_new_black = aug_mask_np[new_black_areas]
        unique_at_new_black = np.unique(mask_values_at_new_black)
        
        if -1 in unique_at_new_black and len(unique_at_new_black) == 1:
            print("✅ PERFECT: All new black areas have -1 (NaN) values!")
        elif len(unique_at_new_black) == 0:
            print("✅ PERFECT: All new black areas have NaN values!")
        else:
            print(f"❌ PROBLEM: New black areas have mixed values: {unique_at_new_black}")
    else:
        print("ℹ️  No new black areas created by augmentation")
    
    # Rest of your visualization code stays the same...
    unique_mask_values = sorted(np.unique(aug_mask_np[~np.isnan(aug_mask_np)]))
    print(f"Final unique mask values (excluding NaN): {unique_mask_values}")
    
    # 🎯 COLOR MAPPING - Include -1 as white and handle NaN
    color_mapping = {
        -1: '#FFFFFF',  # NaN/No-data (converted from NaN) - White
        0: '#dae22f',   # BE - Yellow-green  
        1: '#6332ea',   # NPV - Purple
        2: '#e346ee',   # PV - Magenta
        3: '#6da4d4',   # SI - Light blue
        4: '#68e8d3'    # WI - Cyan
    }
    
    label_mapping = {
        -1: 'NaN',
        0: 'BE',
        1: 'NPV', 
        2: 'PV',
        3: 'SI',
        4: 'WI'
    }
    
    # Handle NaN in display mask
    display_mask_values = unique_mask_values.copy()
    if num_nan > 0:
        display_mask_values = [-1] + display_mask_values  # Add NaN as -1 for display
    
    # Build colors and labels based on what exists
    class_colors = []
    class_labels = []
    
    for mask_value in display_mask_values:
        if mask_value in color_mapping:
            class_colors.append(color_mapping[mask_value])
            class_labels.append(label_mapping[mask_value])
        else:
            class_colors.append('#808080')  # Gray
            class_labels.append(f'Class {int(mask_value)}')
    
    cmap = ListedColormap(class_colors)
    
    # Create display mask (convert NaN to -1 for visualization)
    display_mask = aug_mask_np.copy()
    if num_nan > 0:
        display_mask = np.where(np.isnan(aug_mask_np), -1, aug_mask_np)
    
    # Map to display indices for colormap
    final_display_mask = np.zeros_like(display_mask, dtype=int)
    for i, mask_value in enumerate(display_mask_values):
        final_display_mask[display_mask == mask_value] = i
    
    # Rest of visualization code (unchanged)
    def create_rgb_composite(image_np):
        rgb = None
        if image_np.shape[0] >= 5:
            band_indices = [4, 2, 0]
            rgb = np.zeros((image_np.shape[1], image_np.shape[2], 3))
            for i, band_idx in enumerate(band_indices):
                band = image_np[band_idx].copy()
                p2 = np.percentile(band, 2)
                p98 = np.percentile(band, 98)
                rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        elif image_np.shape[0] >= 3:
            rgb = image_np[:3].transpose(1, 2, 0).copy()
            for i in range(3):
                band = rgb[:,:,i]
                p2 = np.percentile(band, 2)
                p98 = np.percentile(band, 98)
                rgb[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        return rgb
    
    orig_rgb = create_rgb_composite(orig_image_np)
    aug_rgb = create_rgb_composite(aug_image_np)
    
    plt.figure(figsize=(20, 6))
    
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

    # Create a white background first
    plt.imshow(np.ones_like(aug_mask_np), cmap='gray', vmin=0, vmax=1)

    # Get the class values (excluding NaN)
    unique_mask_values = sorted(np.unique(aug_mask_np[~np.isnan(aug_mask_np)]))

    # Define your original color palette
    class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # Original class colors
    cmap = ListedColormap(class_colors[:len(unique_mask_values)])

    # Create a masked array to handle NaNs properly
    masked_display = np.ma.masked_array(
        aug_mask_np, 
        mask=np.isnan(aug_mask_np)  # Mask where values are NaN
    )

    # Plot the mask with NaN areas showing the white background
    im = plt.imshow(masked_display, interpolation='nearest', cmap=cmap,
                    vmin=0, vmax=len(unique_mask_values)-1)

    # Optional hatching for NaN areas for better visibility
    nan_mask = np.isnan(aug_mask_np)
    if np.any(nan_mask):
        plt.contourf(nan_mask, hatches=['//'], colors='none', alpha=0.15)

    plt.title(f"Augmented Mask ({len(unique_mask_values)} classes + NaN)")
    plt.axis('off')

    # Colorbar showing the actual classes
    colorbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
    colorbar.set_ticks(range(len(unique_mask_values)))
    class_labels = ['BE', 'NPV', 'PV', 'SI', 'WI'][:len(unique_mask_values)]
    colorbar.ax.set_yticklabels(class_labels, fontsize=10)
    
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
#             aug_image, aug_mask = albumentations_wrapper(image_tensor, mask_tensor)
#             aug_image_np = aug_image.numpy()
#             aug_mask_np = aug_mask.numpy()
            
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
    # a) Create augmentation wrapper - it will automatically use get_train_augmentation()
    albumentations_transform = get_train_augmentation() 
    albumentations_wrapper = AlbumentationsTorchWrapper(albumentations_transform)


    # 3. Decide how many augmentations per sample
    NUM_AUG_PER_IMAGE = 2
    
    # 🎯 TESTING: Process only first 5 images
    NUM_TEST_IMAGES = 5
    total_images = min(NUM_TEST_IMAGES, len(dataset))
    
    print(f"🧪 TESTING MODE: Processing only {total_images} images out of {len(dataset)} total")

    # 4. Iterate and generate augmentations
    print("Generating and saving augmented data...")
    orig_img_dir = image_dirs[0]
    orig_mask_dir = mask_dirs[0]
    orig_img_files = sorted([f for f in os.listdir(orig_img_dir) if f.endswith('.tif')])
    orig_mask_files = sorted([f for f in os.listdir(orig_mask_dir) if f.endswith('.tif')])

    # 🎯 MODIFIED: Only process first 5 images
    for idx in tqdm(range(total_images), desc="Processing test images"):
        image_tensor, mask_tensor = dataset[idx]
        
        # Debug: Check loaded tensor values
        print(f"Loaded tensor {idx}: min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}")
        
        # Convert to numpy for debug visualization
        image_np = image_tensor.numpy()
        print(f"Original image {idx}: min={image_np.min():.4f}, max={image_np.max():.4f}, mean={image_np.mean():.4f}")
        mask_np = mask_tensor.numpy()

        orig_img_path = os.path.join(orig_img_dir, orig_img_files[idx])
        orig_mask_path = os.path.join(orig_mask_dir, orig_mask_files[idx])

        for aug_idx in tqdm(range(1, NUM_AUG_PER_IMAGE+1), 
                    desc=f"Augmenting image {idx}", 
                    leave=False):
                    
            # Apply augmentation - NOW RETURNS BOTH TENSOR AND RAW NUMPY
            (aug_image_tensor, aug_mask_tensor), (aug_image_np, aug_mask_np) = albumentations_wrapper(image_tensor, mask_tensor)
            
            # Print stats
            print(f"  Aug {aug_idx}: min={aug_image_np.min():.4f}, max={aug_image_np.max():.4f}, mean={aug_image_np.mean():.4f}")
            
            # Use raw numpy arrays with NaN for debug and saving
            debug_visualize(image_np, aug_image_np, aug_mask_np, f"{idx}_{aug_idx}")
            
            # Save using raw numpy arrays (with NaN preserved)
            save_augmented_pair(orig_img_path, orig_mask_path, aug_image_np, aug_mask_np, aug_idx, output_dir_img, output_dir_mask)
    
    # 📊 Summary
    print(f"\n✅ TESTING COMPLETE!")
    print(f"   - Processed: {total_images} original images")
    print(f"   - Generated: {total_images * NUM_AUG_PER_IMAGE} augmented images") 
    print(f"   - Debug visualizations: {total_images * NUM_AUG_PER_IMAGE} PNG files")
    print(f"   - Total files created: {total_images * NUM_AUG_PER_IMAGE * 2} (images + masks)")

if __name__ == "__main__":
    main()