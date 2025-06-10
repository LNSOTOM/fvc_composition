import os
import torch
from tqdm import tqdm
import numpy as np
import rasterio

from dataset.calperum_dataset import CalperumDataset
from dataset.data_augmentation_wrapper import AlbumentationsTorchWrapper
from dataset.data_augmentation import get_train_augmentation, save_augmented_pair
import config_param

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
    NUM_AUG_PER_IMAGE = 3

    # 4. Iterate and generate augmentations
    print("Generating and saving augmented data...")
    orig_img_dir = image_dirs[0]
    orig_mask_dir = mask_dirs[0]
    orig_img_files = sorted([f for f in os.listdir(orig_img_dir) if f.endswith('.tif')])
    orig_mask_files = sorted([f for f in os.listdir(orig_mask_dir) if f.endswith('.tif')])

    for idx in tqdm(range(len(dataset))):
        image_tensor, mask_tensor = dataset[idx]
        # Convert to numpy for rasterio saving
        # [C,H,W] -> rasterio expects [C,H,W] for multiband, mask [H,W]
        image_np = image_tensor.numpy()
        mask_np = mask_tensor.numpy()

        orig_img_path = os.path.join(orig_img_dir, orig_img_files[idx])
        orig_mask_path = os.path.join(orig_mask_dir, orig_mask_files[idx])

        for aug_idx in range(1, NUM_AUG_PER_IMAGE+1):
            # Apply augmentation
            aug_image, aug_mask = albumentations_wrapper(image_tensor, mask_tensor)
            aug_image_np = aug_image.numpy()
            aug_mask_np = aug_mask.numpy()
            # Save
            save_augmented_pair(orig_img_path, orig_mask_path, aug_image_np, aug_mask_np, aug_idx, output_dir_img, output_dir_mask)

if __name__ == "__main__":
    main()