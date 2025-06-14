import os
import torch
from torch.utils.data import Dataset
import numpy as np

from dataset.image_preprocessing import load_raw_multispectral_image, convertImg_to_tensor
from dataset.mask_preprocessing import prep_mask, convertMask_to_tensor, prep_mask_preserve_nan

class CalperumDataset(Dataset):
    def __init__(self, image_folders=None, mask_folders=None, transform=None, in_memory_data=None,
                 save_augmented=False, augmented_save_dir=None):
        self.save_augmented = save_augmented
        self.augmented_save_dir = augmented_save_dir
        self.transform = transform

        if in_memory_data is not None:
            self.images, self.masks = in_memory_data
            if len(self.images) != len(self.masks):
                raise ValueError("The number of images and masks must be the same.")
            return

        if isinstance(image_folders, str):
            image_folders = [image_folders]
        if isinstance(mask_folders, str):
            mask_folders = [mask_folders]

        if not isinstance(image_folders, list) or not isinstance(mask_folders, list):
            raise TypeError("image_folders and mask_folders should be lists of strings or single string paths.")

        self.image_filenames = []
        for image_folder, mask_folder in zip(image_folders, mask_folders):
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
            mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.tif')])
            mask_dict = {os.path.basename(f): f for f in mask_files}

            for image_file in image_files:
                base = image_file.split("tiles_multispectral_")[-1]
                prefix = image_file.split("tiles_multispectral_")[0]
                if prefix.startswith("aug"):
                    mask_file = prefix + "mask_tiles_multispectral_" + base
                else:
                    mask_file = "mask_tiles_multispectral_" + base

                if mask_file in mask_dict:
                    self.image_filenames.append(
                        (os.path.join(image_folder, image_file), os.path.join(mask_folder, mask_file))
                    )
                else:
                    print(f"Warning: No matching mask found for image {image_file} in {mask_folder}")

        print(f"Loaded {len(self.image_filenames)} image/mask pairs from {image_folders}, {mask_folders}")

    def __getitem__(self, idx):
        if hasattr(self, 'images'):
            image = self.images[idx]
            mask = self.masks[idx]
        else:
            if idx >= len(self.image_filenames):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self.image_filenames)}")

            img_filename, mask_filename = self.image_filenames[idx]
            image, _ = load_raw_multispectral_image(img_filename)

            is_augmented = 'aug' in os.path.basename(img_filename).lower()
            if is_augmented:
                mask, _ = prep_mask_preserve_nan(mask_filename)
                if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.floating):
                    mask = np.where(np.isnan(mask), -1, mask)
            else:
                mask, _ = prep_mask(mask_filename)

        if len(image.shape) != 3:
            raise ValueError(f"Expected image shape [C, H, W], got {image.shape}")
        if len(mask.shape) != 2:
            if len(mask.shape) == 3:
                mask = mask[0]
            else:
                raise ValueError(f"Expected mask shape [H, W], got {mask.shape}")

        image_tensor = image if isinstance(image, torch.Tensor) else torch.from_numpy(image).float()
        if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.floating):
            mask = np.where(np.isnan(mask), -1, mask)
        mask_tensor = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).float()

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        image_tensor = torch.nan_to_num(image_tensor, nan=0.0)
        mask_tensor = torch.nan_to_num(mask_tensor, nan=-1)

        mask_tensor = mask_tensor.long()

        # Clean up
        del image, mask
        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.images) if hasattr(self, 'images') else len(self.image_filenames)

    @staticmethod
    def load_mask(mask_path):
        return prep_mask(mask_path)

    @staticmethod
    def load_subsampled_data(image_subsample_dir, mask_subsample_dir, transform=None):
        images = []
        masks = []

        if isinstance(image_subsample_dir, (str, os.PathLike)):
            image_subsample_dir = [image_subsample_dir]
        if isinstance(mask_subsample_dir, (str, os.PathLike)):
            mask_subsample_dir = [mask_subsample_dir]

        for img_dir, mask_dir in zip(image_subsample_dir, mask_subsample_dir):
            image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

            if len(image_files) != len(mask_files):
                raise ValueError("Mismatch between the number of subsampled images and masks in directories.")

            for img_file, mask_file in zip(image_files, mask_files):
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                is_augmented = 'aug' in os.path.basename(img_file).lower()

                image, _ = load_raw_multispectral_image(img_path)
                if is_augmented:
                    mask, _ = prep_mask_preserve_nan(mask_path)
                    if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.floating):
                        mask = np.where(np.isnan(mask), -1, mask)
                else:
                    mask, _ = prep_mask(mask_path)

                image_tensor = image if isinstance(image, torch.Tensor) else torch.from_numpy(image).float()
                mask_tensor = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).float()

                if transform is not None:
                    image_tensor, mask_tensor = transform((image_tensor, mask_tensor))

                image_tensor = torch.nan_to_num(image_tensor, nan=0.0)
                mask_tensor = torch.nan_to_num(mask_tensor, nan=-1).long()

                images.append(image_tensor)
                masks.append(mask_tensor)

        return images, masks
