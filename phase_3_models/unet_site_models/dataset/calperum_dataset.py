import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import GroupKFold
from PIL import Image
from torchvision import transforms
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import rasterio
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import time

import config_param

from torchvision.utils import save_image

from dataset.image_preprocessing import load_raw_multispectral_image, prep_normalise_image, prep_contrast_stretch_image, convertImg_to_tensor, load_raw_rgb_image
from dataset.mask_preprocessing import prep_mask, prep_mask_preserve_nan, convertMask_to_tensor

from dataset.threshold_be_subsampling import subsample_tiles, estimate_class_frequencies

from dataset.data_augmentation_wrapper import AlbumentationsTorchWrapper
from dataset.data_augmentation import get_train_augmentation, get_val_augmentation


# 2.1 Dataset Handling
class CalperumDataset(Dataset):
    def __init__(self, image_folders=None, mask_folders=None, transform=None, in_memory_data=None, save_augmented=False, augmented_save_dir=None):
        self.save_augmented = save_augmented
        self.augmented_save_dir = augmented_save_dir
        
        if in_memory_data is not None:
            self.images, self.masks = in_memory_data
            if len(self.images) != len(self.masks):
                raise ValueError("The number of images and masks must be the same.")
        else:
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

                # Build a dict for fast mask lookup
                mask_dict = {os.path.basename(f): f for f in mask_files}

                for image_file in image_files:
                    # For original: tiles_multispectral_11868.tif -> mask_tiles_multispectral_11868.tif
                    # For augmented: tiles_multispectral_11868_aug1.tif -> mask_tiles_multispectral_11868_aug1.tif
                    if image_file.startswith("tiles_multispectral_"):
                        mask_candidate = "mask_" + image_file
                    else:
                        # fallback, just prepend mask_ to whatever image_file is
                        mask_candidate = "mask_" + image_file
                    
                    mask_file = mask_dict.get(mask_candidate, None)
                    
                    if mask_file is None:
                        # fallback: try matching by unique number suffix
                        image_num = image_file.split("_")[-1]
                        for candidate in mask_files:
                            if candidate.endswith(image_num):
                                mask_file = candidate
                                break

                    if mask_file is not None:
                        self.image_filenames.append(
                            (os.path.join(image_folder, image_file), os.path.join(mask_folder, mask_file))
                        )
                    else:
                        print(f"Warning: No matching mask found for image {image_file} in {mask_folder}")

            print(f"Loaded {len(self.image_filenames)} image/mask pairs from {image_folders}, {mask_folders}")
            if len(self.image_filenames) < 10:
                print("Sample pairs:", self.image_filenames)

        self.transform = transform

    def __getitem__(self, idx):     
        if hasattr(self, 'images'):
            image = self.images[idx]
            mask = self.masks[idx]
        else:
            if idx >= len(self.image_filenames):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self.image_filenames)}")

            img_filename, mask_filename = self.image_filenames[idx]
            image, _ = load_raw_multispectral_image(img_filename)
            mask, _ = prep_mask(mask_filename)

        image_tensor = convertImg_to_tensor(image, dtype=torch.float32)
        mask_tensor = convertMask_to_tensor(mask, dtype=torch.long)
        
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform((image_tensor, mask_tensor))
        
        return image_tensor, mask_tensor

    def __len__(self):
        if hasattr(self, 'images'):
            return len(self.images)
        return len(self.image_filenames)
    
    @staticmethod
    def load_mask(mask_path):
        mask = prep_mask(mask_path)
        return mask

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

                image, _ = load_raw_multispectral_image(img_path)
                mask, _ = prep_mask(mask_path)

                image_tensor = convertImg_to_tensor(image, dtype=torch.float32)
                mask_tensor = convertMask_to_tensor(mask, dtype=torch.long)

                if transform is not None:
                    image_tensor, mask_tensor = transform((image_tensor, mask_tensor))

                images.append(image_tensor)
                masks.append(mask_tensor)

        return images, masks