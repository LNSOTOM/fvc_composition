import torch
import gc
from torch.utils.data import ConcatDataset
from dataset.data_augmentation import apply_combined_augmentations
import numpy as np
import torch
from dataset.image_preprocessing import load_raw_multispectral_image, prep_normalise_image, prep_contrast_stretch_image, convertImg_to_tensor, load_raw_rgb_image
from dataset.mask_preprocessing import prep_mask, prep_mask_preserve_nan, convertMask_to_tensor
import rasterio
from albumentations import Compose
from dataset.data_augmentation import get_train_augmentation

import numpy as np
import torch
from dataset.data_augmentation import get_train_augmentation

import torch
import gc
from torch.utils.data import ConcatDataset
from dataset.data_augmentation import apply_combined_augmentations
import numpy as np
import torch
from dataset.image_preprocessing import load_raw_multispectral_image, prep_normalise_image, prep_contrast_stretch_image, convertImg_to_tensor, load_raw_rgb_image
from dataset.mask_preprocessing import prep_mask, prep_mask_preserve_nan, convertMask_to_tensor
import rasterio
from albumentations import Compose
from dataset.data_augmentation import get_train_augmentation


class AlbumentationsTorchWrapper:
    """
    Wrapper to apply Albumentations transforms to PyTorch tensors.

    Handles:
    - Conversion between torch tensors and numpy arrays
    - Safe handling of NaNs in masks
    - GPU compatibility
    """

    def __init__(self, transform=None):
        self.transform = transform if transform else get_train_augmentation()

    def __call__(self, *args):
        """
        Args:
            args: Either (image_tensor, mask_tensor) or a tuple containing both.

        Returns:
            Tuple of transformed (image_tensor, mask_tensor)
        """
        # Unpack inputs
        if len(args) == 1 and isinstance(args[0], tuple):
            image_tensor, mask_tensor = args[0]
        elif len(args) == 2:
            image_tensor, mask_tensor = args
        else:
            raise ValueError("Expected (image_tensor, mask_tensor) or a tuple containing both.")

        # Save original dtypes
        orig_img_dtype = image_tensor.dtype
        orig_mask_dtype = mask_tensor.dtype

        # Move to CPU and convert to numpy
        image_np = image_tensor.detach().cpu().numpy()
        mask_np = mask_tensor.detach().cpu().numpy()

        # Convert image to [H, W, C] for Albumentations
        if image_np.ndim == 3 and image_np.shape[0] > 1:
            image_np = np.transpose(image_np, (1, 2, 0))

        try:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_np = transformed['image']
            mask_np = transformed['mask']

            # Convert NaNs to -1 for mask
            if np.isnan(mask_np).any():
                mask_np = np.where(np.isnan(mask_np), -1, mask_np)

        except Exception as e:
            print(f"❌ Augmentation failed: {e}")
            print("➡️ Using original data instead.")
            # Fallback to original data
            if image_np.ndim == 3 and image_np.shape[2] > 1:
                image_np = np.transpose(image_np, (2, 0, 1))
            return image_tensor, mask_tensor

        # Convert image back to [C, H, W] format if needed
        if image_np.ndim == 3 and image_np.shape[2] > 1:
            image_np = np.transpose(image_np, (2, 0, 1))

        # Convert back to torch tensors
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_np)).to(dtype=orig_img_dtype)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_np.astype(int))).to(dtype=orig_mask_dtype)

        return image_tensor, mask_tensor

# class AlbumentationsTorchWrapper:
#     """
#     Wrapper for Albumentations transforms to work with PyTorch tensors and preserve NaN values.
#     NaN mask pixels are reassigned using the augmented image’s zero regions.
#     """

#     def __init__(self, transform=None):
#         self.transform = transform if transform is not None else get_train_augmentation()

#     def __call__(self, image_tensor, mask_tensor):
#         # Ensure inputs are torch tensors and convert to numpy
#         if isinstance(image_tensor, torch.Tensor):
#             image_np = image_tensor.cpu().numpy()
#         elif isinstance(image_tensor, np.ndarray):
#             image_np = image_tensor
#         else:
#             raise TypeError(f"Unsupported image type: {type(image_tensor)}")

#         if isinstance(mask_tensor, torch.Tensor):
#             mask_np = mask_tensor.cpu().numpy()
#         elif isinstance(mask_tensor, np.ndarray):
#             mask_np = mask_tensor
#         else:
#             raise TypeError(f"Unsupported mask type: {type(mask_tensor)}")

#         # Ensure float32 for NaN support
#         if not np.issubdtype(mask_np.dtype, np.floating):
#             mask_np = mask_np.astype(np.float32)

#         try:
#             # Apply Albumentations transform
#             transformed = self.transform(
#                 image=image_np.transpose(1, 2, 0),  # CHW -> HWC
#                 mask=mask_np
#             )
#             aug_image_np = transformed['image'].transpose(2, 0, 1)  # HWC -> CHW
#             aug_mask_np = transformed['mask']

#             # Identify NaN areas using zero-valued pixels in the image
#             nan_mask = np.all(aug_image_np == 0, axis=0)
#             aug_mask_np[nan_mask] = np.nan

#             # Ensure float32 for NaNs
#             if not np.issubdtype(aug_mask_np.dtype, np.floating):
#                 aug_mask_np = aug_mask_np.astype(np.float32)

#             # Convert back to tensors
#             aug_image_tensor = torch.from_numpy(aug_image_np).float()
#             aug_mask_tensor = torch.from_numpy(aug_mask_np).float()  # Still float, will be cast to long later

#             return aug_image_tensor, aug_mask_tensor

#         except Exception as e:
#             print(f"❌ Albumentations augmentation failed: {e}")
#             # Fallback to original inputs
#             image_tensor = torch.from_numpy(image_np).float() if isinstance(image_tensor, np.ndarray) else image_tensor
#             mask_tensor = torch.from_numpy(mask_np).float() if isinstance(mask_tensor, np.ndarray) else mask_tensor
#             return image_tensor, mask_tensor

    

class MemoryEfficientAugmentation(torch.utils.data.Dataset):
    """Memory-efficient augmentation that doesn't duplicate the dataset in memory.
    
    This implementation:
    1. Creates virtual augmented samples without storing them in memory
    2. Maintains original indices information for spatial cross-validation
    3. Supports configurable augmentation ratio
    4. Preserves all attributes from the original dataset
    """
    def __init__(self, base_dataset, indices=None, augmentation_ratio=1.0):
        self.dataset = base_dataset
        self.indices = indices if indices is not None else list(range(len(base_dataset)))
        self.augmentation_ratio = augmentation_ratio
        
        # Calculate effective length based on original dataset and augmentation ratio
        self.effective_len = int(len(self.indices) * (1 + augmentation_ratio))
        
        # Store original indices for reference in water distribution
        self.original_indices = self.indices.copy() if hasattr(self.indices, 'copy') else self.indices
    
    def __len__(self):
        return self.effective_len
    
    def __getitem__(self, idx):
        # If idx is beyond original dataset length, it's an augmented sample
        is_augmented = idx >= len(self.indices)
        
        # Get the actual index in the original dataset
        real_idx = idx % len(self.indices)
        original_idx = self.indices[real_idx]
        
        # Get the original sample
        image, mask = self.dataset[original_idx]
        
        # Apply augmentation only to augmented indices
        if is_augmented:
            image, mask = apply_combined_augmentations(image, mask)
            
        return image, mask
        
    def __getattr__(self, name):
        # Pass through any attributes not found to the underlying dataset
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_original_sample(self, idx):
        """Get the original sample without augmentation, useful for debugging."""
        if idx >= len(self.indices):
            real_idx = idx % len(self.indices)
            original_idx = self.indices[real_idx]
            return self.dataset[original_idx]
        return self.dataset[self.indices[idx]]

# Keep this class for backward compatibility
class AugmentationWrapper(MemoryEfficientAugmentation):
    """Legacy wrapper for backward compatibility."""
    def __init__(self, dataset):
        super().__init__(dataset, indices=None, augmentation_ratio=1.0)

# Enhanced version of IndexedConcatDataset with memory management
class IndexedConcatDataset(ConcatDataset):
    """ConcatDataset that preserves dataset indices information with memory optimization."""
    def __init__(self, datasets, indices=None, original_indices=None):
        super().__init__(datasets)
        self.indices = indices
        self.original_indices = original_indices
        
    def __getitem__(self, idx):
        # Get the item using parent implementation
        result = super().__getitem__(idx)
        
        # Explicitly release any intermediate variables to help garbage collection
        if idx % 50 == 0:  # Periodically trigger GC to prevent memory buildup
            gc.collect()
            
        return result
    
from torch.utils.data import Subset

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        self.original_indices = list(indices)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[self.indices[idx]]
        
        # Add debug print statement
        if self.transform:
            print(f"TransformSubset applying transform to item {idx}")
            image, mask = self.transform((image, mask))
            
        return image, mask
    
from torch.utils.data import Dataset, Subset

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            # Ensure proper unpacking
            if isinstance(sample, tuple) and len(sample) == 2:
                image, mask = sample
                print(f"🔄 SubsetWithTransform applying transform to item {index}")
                image, mask = self.transform(image, mask)  # Call with separate arguments
                return image, mask
            else:
                return self.transform(sample)
        return sample

    def __len__(self):
        return len(self.subset)
    
    @property
    def transform(self):
        return getattr(self, '_transform', None)
    
    @transform.setter
    def transform(self, value):
        self._transform = value