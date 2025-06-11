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
    """Wrapper for albumentations transforms to work with PyTorch tensors and preserve NaN values"""
    
    def __init__(self, transform=None):
        if transform is None:
            from dataset.data_augmentation import get_train_augmentation
            self.transform = get_train_augmentation()
        else:
            self.transform = transform
            
    def __call__(self, image_tensor, mask_tensor):
        """Apply albumentations transforms to inputs (as numpy arrays) and preserve no‐data (NaN) pixels in the mask."""
        import numpy as np
        import torch

        # Convert inputs to numpy arrays if needed
        image_np = image_tensor if not isinstance(image_tensor, torch.Tensor) else image_tensor.numpy()
        mask_np  = mask_tensor  if not isinstance(mask_tensor, torch.Tensor) else mask_tensor.numpy()

        # Ensure mask is float32 (so it can represent NaN)
        if not np.issubdtype(mask_np.dtype, np.floating):
            mask_np = mask_np.astype(np.float32)

        # Create a binary nan-mask: 1 if NaN, 0 if valid
        nan_mask = np.isnan(mask_np).astype(np.uint8)

        # Debug: Pre-transform info
        num_nan_before = np.sum(nan_mask)
        print(f"📊 PRE-AUG: mask shape={mask_np.shape}, dtype={mask_np.dtype}")
        print(f"  - Original NaN count: {num_nan_before}")
        print(f"  - Unique valid values: {np.unique(mask_np[~np.isnan(mask_np)])}")

        try:
            # Transform the image and mask as usual
            transformed = self.transform(
                image = image_np.transpose(1, 2, 0),  # [C, H, W] -> [H, W, C]
                mask  = mask_np                        # [H, W]
            )
            aug_image_np = transformed['image'].transpose(2, 0, 1)  # Back to [C, H, W]
            aug_mask_np  = transformed['mask']

            # Also transform the nan_mask (which marks original no-data regions)
            transformed_nan = self.transform(
                image = image_np.transpose(1, 2, 0),
                mask  = nan_mask      # [H, W]
            )
            aug_nan_mask = transformed_nan['mask']
            # Ideally, aug_nan_mask remains 1 where originally NaN (even after interpolation)

            # Ensure aug_mask_np is float32 so it can carry NaN
            if not np.issubdtype(aug_mask_np.dtype, np.floating):
                aug_mask_np = aug_mask_np.astype(np.float32)

            # Now force all pixels that correspond to augmented nan regions to NaN.
            aug_mask_np[aug_nan_mask == 1] = np.nan

            # Debug: Post-transform info
            num_nan_after = np.sum(np.isnan(aug_mask_np))
            print(f"📊 POST-AUG: mask shape={aug_mask_np.shape}, dtype={aug_mask_np.dtype}")
            print(f"  - NaN count after aug: {num_nan_after}")
            print(f"  - Min value (excl. NaN): {np.nanmin(aug_mask_np)}")
            print(f"  - Unique valid values: {np.unique(aug_mask_np[~np.isnan(aug_mask_np)])}")

            # Return NumPy arrays for visualization/saving.
            if isinstance(image_tensor, torch.Tensor):
                aug_image_tensor = torch.tensor(aug_image_np, dtype=torch.float32)
                aug_mask_tensor  = torch.tensor(aug_mask_np, dtype=torch.float32)
                return (aug_image_tensor, aug_mask_tensor), (aug_image_np, aug_mask_np)
            else:
                return (aug_image_np, aug_mask_np)

        except Exception as e:
            print(f"❌ Augmentation error: {e}")
            if isinstance(image_tensor, torch.Tensor):
                return (image_tensor, mask_tensor), (image_np, mask_np)
            else:
                return (image_np, mask_np)
    

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