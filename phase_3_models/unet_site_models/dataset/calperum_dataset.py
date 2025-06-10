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


# 2.1 Dataset Handling
class CalperumDataset(Dataset):
    def __init__(self, image_folders=None, mask_folders=None, transform=None, in_memory_data=None, save_augmented=False, augmented_save_dir=None):
        self.save_augmented = save_augmented
        self.augmented_save_dir = augmented_save_dir
        # self.return_profiles = return_profiles
        
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
                for image_file in image_files:
                    self.image_filenames.append((os.path.join(image_folder, image_file), os.path.join(mask_folder, 'mask_' + image_file)))

        self.transform = transform

    def __getitem__(self, idx):     
        if hasattr(self, 'images'):
            image = self.images[idx]
            mask = self.masks[idx]
            # img_profile, mask_profile = None, None  # Profiles not available in memory mode
        else:
            if idx >= len(self.image_filenames):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self.image_filenames)}")

            img_filename, mask_filename = self.image_filenames[idx]
            
            # Load raw multispectral data using the newly defined function
            image, _ = load_raw_multispectral_image(img_filename)
            ## Use the first element from the tuple returned by prep_normalise_image
            # image, _ = prep_normalise_image(img_filename)           
            # # Similarly, for contrast stretching, if used:
            # image, _ = prep_contrast_stretch_image(img_filename)    
            ## Load raw RGB
            # image, _ = load_raw_rgb_image(img_filename)      
            
            mask, _ = prep_mask(mask_filename)

        # Convert numpy arrays to tensors
        image_tensor = convertImg_to_tensor(image, dtype=torch.float32) # MULTISPECTRAL
        # image_tensor = convertImg_to_tensor(image, dtype=torch.uint8)  # RGB
        mask_tensor = convertMask_to_tensor(mask, dtype=torch.long)
        
        # Apply transformations to both image and mask tensors
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform((image_tensor, mask_tensor))
        
        return image_tensor, mask_tensor
        # ## Return only tensors if return_profiles is False
        # if self.return_profiles:
        #     return image_tensor, mask_tensor, img_profile, mask_profile
        # else:
        #     return image_tensor, mask_tensor
          

    def __len__(self):
        if hasattr(self, 'images'):
            return len(self.images)
        return len(self.image_filenames)
    
    @staticmethod
    def load_mask(mask_path):
        """Load a mask from the given file path."""
        mask = prep_mask(mask_path)
        return mask
    
    # @staticmethod
    # def load_subsampled_data(image_subsample_dir, mask_subsample_dir, transform=None):
    #     images = []
    #     masks = []

    #     image_files = sorted([f for f in os.listdir(image_subsample_dir) if f.endswith('.tif')])
    #     mask_files = sorted([f for f in os.listdir(mask_subsample_dir) if f.endswith('.tif')])

    #     if len(image_files) != len(mask_files):
    #         raise ValueError("Mismatch between the number of subsampled images and masks.")

    #     for img_file, mask_file in zip(image_files, mask_files):
    #         img_path = os.path.join(image_subsample_dir, img_file)
    #         mask_path = os.path.join(mask_subsample_dir, mask_file)

    #         # Load raw multispectral data using the same method as in __getitem__
    #         image, _ = load_raw_multispectral_image(img_path)
    #         mask, _ = prep_mask(mask_path)

    #         # Convert the images and masks to tensors
    #         image_tensor = convertImg_to_tensor(image, dtype=torch.float32)
    #         mask_tensor = convertMask_to_tensor(mask, dtype=torch.long)

    #         # Apply transformations if provided
    #         if transform is not None:
    #             image_tensor, mask_tensor = transform((image_tensor, mask_tensor))

    #         images.append(image_tensor)
    #         masks.append(mask_tensor)

    #     return images, masks
    @staticmethod
    def load_subsampled_data(image_subsample_dir, mask_subsample_dir, transform=None):
        images = []
        masks = []

        # Ensure image_subsample_dir and mask_subsample_dir are lists if they are not already
        if isinstance(image_subsample_dir, (str, os.PathLike)):
            image_subsample_dir = [image_subsample_dir]
        if isinstance(mask_subsample_dir, (str, os.PathLike)):
            mask_subsample_dir = [mask_subsample_dir]

        # Iterate over each directory in the lists
        for img_dir, mask_dir in zip(image_subsample_dir, mask_subsample_dir):
            image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

            if len(image_files) != len(mask_files):
                raise ValueError("Mismatch between the number of subsampled images and masks in directories.")

            for img_file, mask_file in zip(image_files, mask_files):
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)

                # Load raw multispectral data
                image, _ = load_raw_multispectral_image(img_path)
                mask, _ = prep_mask(mask_path)

                # Convert images and masks to tensors
                image_tensor = convertImg_to_tensor(image, dtype=torch.float32)
                mask_tensor = convertMask_to_tensor(mask, dtype=torch.long)

                # Apply transformations if provided
                if transform is not None:
                    image_tensor, mask_tensor = transform((image_tensor, mask_tensor))

                images.append(image_tensor)
                masks.append(mask_tensor)

        return images, masks


