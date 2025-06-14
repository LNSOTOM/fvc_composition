import os
import numpy as np
from tqdm import tqdm
from dataset.calperum_dataset import CalperumDataset
from dataset.data_augmentation_wrapper import AlbumentationsTorchWrapper
from dataset.data_augmentation import get_train_augmentation, save_augmented_pair
from dataset.image_preprocessing import load_raw_multispectral_image
from dataset.mask_preprocessing import prep_mask_preserve_nan
import config_param
import shutil

def generate_filtered_augmentations_from_train_only(train_indices, block_idx):
    """
    Generate augmentations only for training indices and save them to disk.
    This avoids data leakage from validation/test data.
    """

    # Setup output directories for this block
    output_dir_img = f'/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/block{block_idx}/predictor_5b'
    output_dir_mask = f'/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/block{block_idx}/mask_fvc'
    
    # Clear old augmentations if they exist
    if os.path.exists(output_dir_img):
        shutil.rmtree(output_dir_img)
    if os.path.exists(output_dir_mask):
        shutil.rmtree(output_dir_mask)
        
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)

    # Load dataset with no transform
    dataset = CalperumDataset(
        image_folders=config_param.SUBSAMPLE_IMAGE_DIR,
        mask_folders=config_param.SUBSAMPLE_MASK_DIR,
        transform=None
    )

    # Set up augmentation transform
    albumentations_transform = get_train_augmentation()
    alb_wrapper = AlbumentationsTorchWrapper(albumentations_transform, debug=False)
    NUM_AUG_PER_IMAGE = 2

    print(f"🔁 Generating {NUM_AUG_PER_IMAGE} augmentations for each of the {len(train_indices)} training samples (Block {block_idx})")

    # Get all file paths directly from the directories
    orig_img_dir = config_param.SUBSAMPLE_IMAGE_DIR[0]
    orig_mask_dir = config_param.SUBSAMPLE_MASK_DIR[0]
    orig_img_files = sorted([f for f in os.listdir(orig_img_dir) if f.endswith('.tif')])
    orig_mask_files = sorted([f for f in os.listdir(orig_mask_dir) if f.endswith('.tif')])

    for count, idx in enumerate(tqdm(train_indices, desc="Augmenting training data")):
        # Get file paths directly using the index
        img_path = os.path.join(orig_img_dir, orig_img_files[idx])
        mask_path = os.path.join(orig_mask_dir, orig_mask_files[idx])

        # Load image/mask as NumPy arrays (NaN-aware)
        image_np, _ = load_raw_multispectral_image(img_path)
        mask_np, _ = prep_mask_preserve_nan(mask_path)

        # Debug info
        if count % 10 == 0:  # Print every 10th image to reduce output
            print(f"[{count}] {os.path.basename(img_path)}")
            print(f" - Image shape: {image_np.shape}, range: ({np.nanmin(image_np):.3f}, {np.nanmax(image_np):.3f})")
            print(f" - Mask shape: {mask_np.shape}, unique: {np.unique(mask_np[~np.isnan(mask_np)])}")

        for aug_idx in range(1, NUM_AUG_PER_IMAGE + 1):
            aug_image_np, aug_mask_np = alb_wrapper(image_np, mask_np)

            # Save augmented pair
            save_augmented_pair(
                img_path, mask_path,
                aug_image_np, aug_mask_np,
                aug_idx,
                output_dir_img, output_dir_mask
            )

    print(f"\n✅ Done augmenting training set for Block {block_idx}")
    print(f"   - Augmentations saved to: {output_dir_img} and {output_dir_mask}")
    print(f"   - Total augmented images created: {len(train_indices) * NUM_AUG_PER_IMAGE}")

