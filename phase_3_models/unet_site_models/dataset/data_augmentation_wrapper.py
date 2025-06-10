import torch
import gc
from torch.utils.data import ConcatDataset
from dataset.data_augmentation import apply_combined_augmentations
import numpy as np
import torch
from dataset.image_preprocessing import load_raw_multispectral_image, prep_normalise_image, prep_contrast_stretch_image, convertImg_to_tensor, load_raw_rgb_image
from dataset.mask_preprocessing import prep_mask, prep_mask_preserve_nan, convertMask_to_tensor
import rasterio

class AlbumentationsTorchWrapper:
    """Wrapper for Albumentations transforms to work with PyTorch tensors."""
    
    def __init__(self, transform):
        self.transform = transform
             
    def __call__(self, *args):
        """Convert tensors to numpy, apply transform, then convert back to tensors with proper types"""
        # Handle both tuple input and separate arguments
        if len(args) == 1 and isinstance(args[0], tuple):
            image_tensor, mask_tensor = args[0]
        elif len(args) == 2:
            image_tensor, mask_tensor = args
        else:
            raise ValueError("Expected either a tuple of (image, mask) or separate image and mask arguments")

        # Store original tensor dtypes for conversion back later
        orig_img_dtype = image_tensor.dtype
        orig_mask_dtype = mask_tensor.dtype
        
        # Convert to numpy for transformations
        image_np = image_tensor.numpy()
        mask_np = mask_tensor.numpy()

        # Debug: Print original values
        print(f"Original image shape: {image_np.shape}, mask shape: {mask_np.shape}")
        print(f"Original image stats: min={image_np.min():.6f}, max={image_np.max():.6f}, mean={image_np.mean():.6f}")
        
        # Check if we need to recover data due to zero values
        if image_np.min() == 0 and image_np.max() == 0:
            print("WARNING: Input image is all zeros! Attempting recovery...")
            
            # Try to access file path if available
            if hasattr(image_tensor, 'file_path') and image_tensor.file_path:
                try:
                    print(f"Recovery attempt using file_path: {image_tensor.file_path}")
                    with rasterio.open(image_tensor.file_path) as src:
                        recovered_image = src.read()
                        print(f"Recovery successful: min={recovered_image.min()}, max={recovered_image.max()}")
                        
                        # Convert recovered image to HWC format for Albumentations
                        if recovered_image.ndim == 3 and recovered_image.shape[0] > 1:
                            image_np = np.transpose(recovered_image, (1, 2, 0))  # [C,H,W] -> [H,W,C]
                        else:
                            image_np = recovered_image
                            
                except Exception as e:
                    print(f"Recovery failed using file_path: {e}")
                    # Create dummy data as absolute fallback
                    if image_np.ndim == 3:
                        image_np = np.random.uniform(0.1, 0.5, size=image_np.shape).astype(np.float32)
                    else:
                        image_np = np.random.uniform(0.1, 0.5, size=(256, 256, 5)).astype(np.float32)
            else:
                print("No file_path available, creating dummy data")
                # Create dummy data as fallback
                if image_np.ndim == 3 and image_np.shape[0] <= 5:
                    # Input is in CHW format, transpose to HWC for augmentation
                    image_np = np.random.uniform(0.1, 0.5, size=(256, 256, 5)).astype(np.float32)
                else:
                    image_np = np.random.uniform(0.1, 0.5, size=image_np.shape).astype(np.float32)
        else:
            # Convert image from NCHW to HWC format for Albumentations
            if image_np.ndim == 3 and image_np.shape[0] > 1:  # If it has multiple channels
                image_np = np.transpose(image_np, (1, 2, 0))  # [C,H,W] -> [H,W,C]
            
        print(f"Transposed image shape: {image_np.shape}, mask shape: {mask_np.shape}")

        # Apply albumentations transform
        transformed = self.transform(image=image_np, mask=mask_np)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        print(f"Transformed image shape: {transformed_image.shape}, mask shape: {transformed_mask.shape}")
        
        # Check for zero values after augmentation and apply fallback if needed
        if transformed_image.min() == 0 and transformed_image.max() == 0:
            print("WARNING: All-zero image after augmentation! Using fallback...")
            # Fallback: Apply simple flip instead
            if np.random.rand() > 0.5:
                transformed_image = np.flip(image_np, axis=1).copy()  # Horizontal flip
            else:
                transformed_image = np.flip(image_np, axis=0).copy()  # Vertical flip
                
            if np.random.rand() > 0.5:
                transformed_mask = np.flip(mask_np, axis=0).copy()
            else:
                transformed_mask = np.flip(mask_np, axis=1).copy()
        
        # Convert back from HWC to NCHW format for PyTorch
        if transformed_image.ndim == 3 and transformed_image.shape[2] > 1:
            transformed_image = np.transpose(transformed_image, (2, 0, 1))  # [H,W,C] -> [C,H,W]
        
        # Ensure arrays are C-contiguous to avoid stride issues
        transformed_image = np.ascontiguousarray(transformed_image)
        transformed_mask = np.ascontiguousarray(transformed_mask)
        
        # Convert directly back to PyTorch tensor with ORIGINAL dtypes
        image_tensor = torch.from_numpy(transformed_image).to(dtype=orig_img_dtype)
        mask_tensor = torch.from_numpy(transformed_mask).to(dtype=orig_mask_dtype)
    
        print(f"Final tensor shapes - image: {image_tensor.shape}, mask: {mask_tensor.shape}")
        print(f"Final image stats: min={image_tensor.min().item():.6f}, max={image_tensor.max().item():.6f}")
        
        return image_tensor, mask_tensor

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