import os
import rasterio
from dataset.image_preprocessing import load_raw_multispectral_image
from dataset.mask_preprocessing import prep_mask
from config_param import IMAGE_FOLDER, MASK_FOLDER, AUG_IMAGE_DIR, AUG_MASK_DIR

def save_augmented_pair(orig_img_path, orig_mask_path, aug_image, aug_mask, aug_idx, aug_img_dir, aug_mask_dir):
    img_name = os.path.basename(orig_img_path).replace('.tif', '')
    mask_name = os.path.basename(orig_mask_path).replace('.tif', '')

    aug_img_name = f"{img_name}_aug{aug_idx}.tif"
    aug_mask_name = f"{mask_name}_aug{aug_idx}.tif"
    aug_img_path = os.path.join(aug_img_dir, aug_img_name)
    aug_mask_path = os.path.join(aug_mask_dir, aug_mask_name)

    # Save image
    with rasterio.open(orig_img_path) as src:
        meta = src.meta.copy()
    with rasterio.open(aug_img_path, 'w', **meta) as dst:
        dst.write(aug_image)

    # Save mask
    with rasterio.open(orig_mask_path) as src:
        meta = src.meta.copy()
    with rasterio.open(aug_mask_path, 'w', **meta) as dst:
        dst.write(aug_mask, 1)

def generate_and_save_augmentations():
    from tqdm import tqdm
    # Loop through all original image/mask pairs
    for img_dir, mask_dir in zip(IMAGE_FOLDER, MASK_FOLDER):
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

        for img_file, mask_file in tqdm(zip(img_files, mask_files), total=len(img_files)):
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # Load image and mask
            image, _ = load_raw_multispectral_image(img_path)
            mask, _ = prep_mask(mask_path)

            # Example: create 3 augmentations per image
            for aug_idx in range(1, 4):
                # Replace this with your actual augmentation
                aug_image = image  # TODO: Apply your augmentation here!
                aug_mask = mask    # TODO: Apply your augmentation here!
                save_augmented_pair(img_path, mask_path, aug_image, aug_mask, aug_idx, AUG_IMAGE_DIR, AUG_MASK_DIR)

if __name__ == '__main__':
    # Uncomment this block to generate augmented data before training
    generate_and_save_augmentations()
