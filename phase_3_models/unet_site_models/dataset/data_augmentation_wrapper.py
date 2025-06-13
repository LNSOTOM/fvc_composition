import torch
import gc
from torch.utils.data import ConcatDataset

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

import numpy as np
import torch
from dataset.image_preprocessing import load_raw_multispectral_image, prep_normalise_image, prep_contrast_stretch_image, convertImg_to_tensor, load_raw_rgb_image
from dataset.mask_preprocessing import prep_mask, prep_mask_preserve_nan, convertMask_to_tensor
import rasterio
from albumentations import Compose
from dataset.data_augmentation import get_train_augmentation


class AlbumentationsTorchWrapper:
    """
    Wrapper to apply Albumentations transforms to PyTorch tensors or NumPy arrays.
    Carefully handles NaN values in masks and preserves no-data regions.
    """
    def __init__(self, transform=None, debug=False):
        self.transform = transform if transform else get_train_augmentation()
        self.debug = debug  # Control debug output

    def __call__(self, *args):
        """
        Apply albumentations transforms to inputs and preserve no-data (NaN) pixels in the mask.
        Uses the augmented image's zero-value pixels as the reference to set no-data regions in the mask.
        
        Args:
            args: Either (image_tensor, mask_tensor) or a tuple containing both.

        Returns:
            Tuple of transformed (image_tensor, mask_tensor)
        """
        # Unpack inputs - standardize to handle both cases
        if isinstance(args, tuple) and len(args) == 2:
            image_tensor, mask_tensor = args
        else:
            image_tensor, mask_tensor = args, None
        
        # Check input types and handle accordingly
        is_torch_input = isinstance(image_tensor, torch.Tensor)
        
        # Skip processing for very large arrays/tensors to avoid OOM
        if is_torch_input:
            if image_tensor.numel() > 5e7:  # Skip for tensors larger than ~50M elements
                return image_tensor, mask_tensor
        else:  # NumPy array
            if image_tensor.size > 5e7:  # Use .size for NumPy arrays
                return image_tensor, mask_tensor
            
        # Save original dtypes
        orig_img_dtype = image_tensor.dtype
        orig_mask_dtype = mask_tensor.dtype if mask_tensor is not None else None

        # Convert inputs to numpy arrays
        if torch.is_tensor(image_tensor):
            image_np = image_tensor.cpu().numpy() if image_tensor.is_cuda else image_tensor.numpy()
        else:
            image_np = np.array(image_tensor)
            
        if mask_tensor is not None:
            if torch.is_tensor(mask_tensor):
                mask_np = mask_tensor.cpu().numpy() if mask_tensor.is_cuda else mask_tensor.numpy()
            else:
                mask_np = np.array(mask_tensor)
                
        # Ensure mask is float32 (so it can represent NaN)
        if not np.issubdtype(mask_np.dtype, np.floating):
            mask_np = mask_np.astype(np.float32)
            
        # Debug: Pre-transform info about NaNs
        num_nan_before = np.sum(np.isnan(mask_np))
        if self.debug and num_nan_before > 0:  # Only print if debug is enabled
            print(f"📊 PRE-AUG: mask shape={mask_np.shape}, dtype={mask_np.dtype}")
            print(f"  - Original NaN count: {num_nan_before}")
            print(f"  - Unique valid values: {np.unique(mask_np[~np.isnan(mask_np)])}")

        # Convert image to [H, W, C] for Albumentations
        if image_np.ndim == 3 and image_np.shape[0] > 1:
            image_np = np.transpose(image_np, (1, 2, 0))

        try:
            # Transform the image and mask
            transformed = self.transform(image=image_np, mask=mask_np)
            aug_image_np = transformed['image']
            aug_mask_np = transformed['mask']
            
            # Convert image back to [C, H, W] format immediately
            if aug_image_np.ndim == 3 and aug_image_np.shape[2] > 1:
                aug_image_np = np.transpose(aug_image_np, (2, 0, 1))
                
            # Use the augmented image as reference to identify no-data regions:
            # 1. Find zero-value pixels across all channels (likely created by transform)
            zero_pixels = np.all(aug_image_np == 0, axis=0)
            
            # 2. Find artificially created edge pixels (where alpha would normally be 0)
            edge_pixels = np.zeros_like(zero_pixels, dtype=bool)
            if np.any(zero_pixels):
                # Dilate the zero regions slightly to catch partial border pixels
                from scipy import ndimage
                edge_pixels = ndimage.binary_dilation(zero_pixels, iterations=1)
            
            # 3. Set mask to NaN in these regions
            aug_mask_np[edge_pixels] = np.nan
            
            # Debug: Post-transform info about NaNs
            num_nan_after = np.sum(np.isnan(aug_mask_np))
            if self.debug and (num_nan_before > 0 or num_nan_after > 0):  # Only print if debug is enabled
                print(f"📊 POST-AUG: mask shape={aug_mask_np.shape}, dtype={aug_mask_np.dtype}")
                print(f"  - NaN count after aug: {num_nan_after}")
                print(f"  - Unique valid values: {np.unique(aug_mask_np[~np.isnan(aug_mask_np)])}")

        except Exception as e:
            if self.debug:  # Only print if debug is enabled
                print(f"❌ Augmentation failed: {e}")
                print("➡️ Using original data instead.")
            
            # Fallback to original data
            if image_np.ndim == 3 and image_np.shape[2] > 1:
                image_np = np.transpose(image_np, (2, 0, 1))
            return image_tensor, mask_tensor

        # If input was a torch tensor, convert back to tensor with correct dtype
        if is_torch_input:
            # Convert NaNs to appropriate value for PyTorch tensors
            if np.isnan(aug_mask_np).any():
                # For training with torch loss functions, use -1 instead of NaN
                aug_mask_np = np.where(np.isnan(aug_mask_np), -1, aug_mask_np)
            
            # Convert to PyTorch tensor with appropriate dtype
            if str(orig_img_dtype) == 'float32':
                image_tensor = torch.from_numpy(np.ascontiguousarray(aug_image_np)).float()
            elif str(orig_img_dtype) == 'float64':
                image_tensor = torch.from_numpy(np.ascontiguousarray(aug_image_np)).double()
            elif str(orig_img_dtype) == 'int64':
                image_tensor = torch.from_numpy(np.ascontiguousarray(aug_image_np)).long()
            elif str(orig_img_dtype) == 'int32':
                image_tensor = torch.from_numpy(np.ascontiguousarray(aug_image_np)).int()
            else:
                image_tensor = torch.from_numpy(np.ascontiguousarray(aug_image_np))

            if mask_tensor is not None:
                if str(orig_mask_dtype) == 'float32':
                    mask_tensor = torch.from_numpy(np.ascontiguousarray(aug_mask_np)).float()
                elif str(orig_mask_dtype) == 'float64':
                    mask_tensor = torch.from_numpy(np.ascontiguousarray(aug_mask_np)).double()
                elif str(orig_mask_dtype) == 'int64':
                    mask_tensor = torch.from_numpy(np.ascontiguousarray(aug_mask_np)).long()
                elif str(orig_mask_dtype) == 'int32':
                    mask_tensor = torch.from_numpy(np.ascontiguousarray(aug_mask_np)).int()
                else:
                    mask_tensor = torch.from_numpy(np.ascontiguousarray(aug_mask_np))
        else:
            # Return NumPy arrays if input was NumPy
            image_tensor = aug_image_np
            mask_tensor = aug_mask_np

        # Clean up to help garbage collection
        del image_np, mask_np, aug_image_np, aug_mask_np
        
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
            # Always pass as a tuple to ensure consistency
            if isinstance(sample, tuple) and len(sample) == 2:
                image, mask = sample
                # Pass as a single tuple argument
                return self.transform((image, mask))
            else:
                return self.transform(sample)
        return sample

    def __len__(self):
        return len(self.subset)