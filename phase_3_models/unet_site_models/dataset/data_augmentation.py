import numpy as np
import torch
import torchvision.transforms as transforms
from collections import Counter
import random
from copy import deepcopy

import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import random
import config_param

import albumentations as A

import os
import rasterio

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

def get_train_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=15, border_mode=0, p=0.4),
        # A.RandomBrightnessContrast(p=0.2),  # Uncomment if you want
        # Add more augmentations as needed
    ])

def get_val_augmentation():
    return A.Compose([
        # Only normalization or resizing if needed
    ])

class RandomAffineTransform:
    def __init__(self, degrees=20, translate=(0.15, 0.15), shear=11.5, p=0.8):
        self.degrees = degrees
        self.translate = translate
        self.shear = shear
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        if random.random() < self.p:
            print("Applied Affine Transform")  # Debugging message
            angle = random.choice(range(-self.degrees, self.degrees + 1, 20))
            translate_pix = [int(t * s) for t, s in zip(self.translate, image.shape[1:])]
            shear_val = random.uniform(-self.shear, self.shear)
            
            # Apply affine transform to image
            image = F.affine(image, angle=angle, translate=translate_pix, scale=1.0, shear=[shear_val], interpolation=InterpolationMode.BILINEAR)
            
            # Add channel dimension to mask, apply transform, then remove channel dimension
            mask = mask.unsqueeze(0)  # Add channel dimension [H,W] -> [1,H,W]
            mask = F.affine(mask, angle=angle, translate=translate_pix, scale=1.0, shear=[shear_val], interpolation=InterpolationMode.NEAREST)
            mask = mask.squeeze(0)  # Remove channel dimension [1,H,W] -> [H,W]
            
        return image, mask

class RandomBrightnessContrast:
    def __init__(self, brightness_range=(0.9, 1.1), contrast_range=(0.8, 1.2), p=0.7):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        if random.random() < self.p:
            brightness_factor = random.uniform(*self.brightness_range)
            contrast_factor = random.uniform(*self.contrast_range)
            for c in range(image.shape[0]):
                channel = image[c]
                channel = channel * brightness_factor
                mean = channel.mean()
                channel = (channel - mean) * contrast_factor + mean
                image[c] = torch.clamp(channel, 0, 1)
        return image, mask

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        if random.random() < self.p:
            print("Applied Vertical Flip")  # Debugging message
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        if random.random() < self.p:
            print("Applied Horizontal Flip")  # Debugging message
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask

def get_transform(train: bool = False, enable_augmentation: bool = False):
    """
    Returns the appropriate transformation pipeline for training or evaluation.
    
    Args:
        train: Whether this is for training (True) or validation/test (False)
        enable_augmentation: Whether to apply data augmentation
    
    Returns:
        A function that applies transformations to (image, mask) pairs
    """
    if train and enable_augmentation:
        def transform_fn(image, mask):
            print("🔄 Applying data augmentation...")  # Debug print
            
            # Apply augmentations (modify probabilities as needed)
            image, mask = apply_color_jitter(image, mask)
            image, mask = apply_vertical_flip(image, mask)
            image, mask = apply_horizontal_flip(image, mask)
            
            return image, mask
        return transform_fn
    else:
        def no_transform(image, mask):
            """Identity transform - returns data unchanged"""
            return image, mask
        return no_transform



###################

def apply_color_jitter(image, mask):
    transform = transforms.ColorJitter(
        brightness=(0.9, 1.1),  # Brightness 90%-110%
        contrast=(0.8, 1.2)     # Contrast 80%-120%
    )
    channels = []
    for i in range(image.shape[0]):
        channel = transforms.ToPILImage()(image[i].unsqueeze(0))
        channel = transform(channel)
        channels.append(transforms.ToTensor()(channel).squeeze(0))
    return torch.stack(channels), mask  # Return the mask unchanged

# Function to apply vertical flip
def apply_vertical_flip(image, mask):
    transform = transforms.RandomVerticalFlip(p=1)
    image = transform(image)
    mask = transform(mask)
    return image, mask

# Function to apply horizontal flip
def apply_horizontal_flip(image, mask):
    transform = transforms.RandomHorizontalFlip(p=1)
    image = transform(image)
    mask = transform(mask)
    return image, mask

# Function to apply random affine transformation
def apply_random_affine(image, mask):
    transform = transforms.RandomAffine(degrees=20, translate=(0.15, 0.15))
    channels = []
    for i in range(image.shape[0]):
        channel = transforms.ToPILImage()(image[i].unsqueeze(0))
        channel = transform(channel)
        channels.append(transforms.ToTensor()(channel).squeeze(0))

    # Apply the same transformation to the mask
    mask = transforms.ToPILImage()(mask)
    mask = transform(mask)
    mask = transforms.ToTensor()(mask)

    return torch.stack(channels), mask


def generate_random_affine_params():
    """Generate random affine transformation parameters."""
    return {
        "angle": random.uniform(-20, 20),
        "translate": [random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)],
        "scale": random.uniform(0.95, 1.05),
        "shear": random.uniform(0, 0.2)
    }

def apply_affine_to_image(image, params):
    """Apply the same affine transformation to all image channels using NEAREST."""
    transformed_channels = []
    for i in range(image.shape[0]):
        pil_channel = transforms.ToPILImage()(image[i].unsqueeze(0))
        transformed = transforms.functional.affine(
            pil_channel,
            angle=params["angle"],
            translate=params["translate"],
            scale=params["scale"],
            shear=params["shear"],
            interpolation=transforms.InterpolationMode.NEAREST
        )
        transformed_channels.append(transforms.ToTensor()(transformed).squeeze(0))
    return torch.stack(transformed_channels)

def apply_affine_to_mask(mask, params):
    """Apply affine transformation with guaranteed class preservation."""
    original_dtype = mask.dtype
    original_unique = torch.unique(mask).tolist()
    h, w = mask.shape
    transformed_mask = torch.ones((h, w), dtype=original_dtype) * -1
    
    # Create a class size dictionary to apply special handling for small classes
    class_sizes = {}
    for class_val in original_unique:
        if class_val >= 0:  # Skip background (-1)
            class_size = (mask == class_val).sum().item()
            class_sizes[class_val] = class_size
            
    # Process classes from smallest to largest for better small class preservation
    sorted_classes = sorted([(cls, size) for cls, size in class_sizes.items()], 
                           key=lambda x: x[1])
    
    # Process each class separately, with special handling for small classes
    for class_val, size in sorted_classes:
        # For very small classes, use a lower threshold to capture more pixels
        threshold = 0.2 if size < 50 else 0.5
        
        # Create a binary mask for this class
        binary_mask = (mask == class_val).to(torch.uint8) * 255
        pil_mask = transforms.ToPILImage()(binary_mask)
        
        # Apply the affine transformation
        transformed = transforms.functional.affine(
            pil_mask,
            angle=params["angle"],
            translate=params["translate"],
            scale=params["scale"],
            shear=params["shear"],
            interpolation=transforms.InterpolationMode.NEAREST
        )
        
        # Convert back to tensor and apply appropriate threshold
        transformed_tensor = transforms.ToTensor()(transformed).squeeze(0)
        binary_result = transformed_tensor > threshold
        
        # If class was completely lost, recover it
        if not binary_result.any() and size > 0:
            # Find center of mass of original class
            indices = torch.nonzero(mask == class_val)
            if len(indices) > 0:
                cy = indices[:, 0].float().mean().round().long()
                cx = indices[:, 1].float().mean().round().long()
                
                # Place a small region in the transformed mask
                for y in range(max(0, cy-1), min(h, cy+2)):
                    for x in range(max(0, cx-1), min(w, cx+2)):
                        transformed_mask[y, x] = class_val
                
                print(f"Recovered lost class {class_val} with {size} pixels")
                continue
                
        # Apply the class value where binary mask is True
        scalar_val = class_val if not isinstance(class_val, torch.Tensor) else class_val.item()
        transformed_mask = torch.where(binary_result, 
                                     torch.tensor(scalar_val, dtype=original_dtype), 
                                     transformed_mask)
    
    # Final verification and recovery of any still-missing classes
    new_unique = torch.unique(transformed_mask).tolist()
    missing = [val for val in original_unique if val >= 0 and val not in new_unique]
    
    if missing:
        for class_val in missing:
            # Force preserve class in corners if all else failed
            corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
            for i, (y, x) in enumerate(corners[:2]):  # Just use 2 corners
                transformed_mask[y, x] = class_val
            print(f"Last-resort recovery of class {class_val} in corners")
    
    return transformed_mask

def apply_combined_augmentations(image, mask):
    """Apply safe augmentations with improved class preservation."""
    original_unique = torch.unique(mask).tolist()
    original_dtype = mask.dtype
    applied = []

    # 1. Random horizontal flip (50% chance)
    if random.random() > 0.5:
        image, mask = apply_horizontal_flip(image, mask)
        applied.append("h_flip")

    # 2. Random vertical flip (50% chance)
    if random.random() > 0.5:
        image, mask = apply_vertical_flip(image, mask)
        applied.append("v_flip")

    # 3. Apply color jitter (50% chance)
    if random.random() > 0.5:
        image, mask = apply_color_jitter(image, mask)
        applied.append("color_jitter")
    
    # 4. Apply affine transformation with reduced probability (40% chance)
    if random.random() > 0.6:
        # Use gentler transformation parameters
        params = {
            "angle": random.uniform(-10, 10),  # Reduced from -20,20
            "translate": [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)],  # Reduced
            "scale": random.uniform(0.95, 1.05),  # More conservative
            "shear": random.uniform(0, 0.1)  # Reduced from 0,0.2
        }
        
        image = apply_affine_to_image(image, params)
        mask = apply_affine_to_mask(mask, params)
        applied.append("affine_nearest")

    # Ensure mask has the right dtype
    mask = mask.to(original_dtype)
    
    # Final verification
    new_unique = torch.unique(mask).tolist()
    missing = {val for val in original_unique if val >= 0 and val not in new_unique}
    if missing:
        print(f"WARNING: Lost classes {missing} after {applied}")
        
    return image, mask



##################
def augment_minority_classes(
    dataset, class_distributions, class_labels, target_ratios,
    fold_assignments=None, augmentation_functions=None, indices_to_augment=None
):
    """
    Augment minority classes in a CalperumDataset to meet target ratios.

    Args:
        dataset: CalperumDataset object with in-memory .images and .masks.
        class_distributions: Dict of current class percentages.
        class_labels: Dict mapping class index to class name.
        target_ratios: Dict mapping class name to target ratio (0-1).
        fold_assignments: Optional dict mapping original sample index to fold.
        augmentation_functions: Optional list of augmentation functions.

    Returns:
        dict: Augmented sample count per class.
    """
    if not hasattr(dataset, "images") or not hasattr(dataset, "masks"):
        raise ValueError("Dataset must be loaded into memory with .images and .masks attributes.")
    if not hasattr(dataset, "augmented_from_idx"):
        dataset.augmented_from_idx = list(range(len(dataset.images)))
    if augmentation_functions is None:
        augmentation_functions = []

    augmented_counts = {}
    new_images = list(dataset.images)
    new_masks = list(dataset.masks)
    new_fold_assignments = dict(fold_assignments) if fold_assignments else {}

    original_size = len(dataset.images)
    for class_idx, class_name in class_labels.items():
        current_ratio = class_distributions.get(class_name, 0)
        target_ratio = target_ratios.get(class_name, 0.1) * 100  # Target in percent

        if current_ratio < target_ratio:
            print(f"Augmenting class '{class_name}' (current ratio: {current_ratio:.2f}%, target ratio: {target_ratio:.2f}%)")

            # Only use training samples for augmentation
            class_samples = [
                (img, mask, idx)
                for idx, (img, mask) in enumerate(zip(dataset.images, dataset.masks))
                if (np.array(mask) == class_idx).sum() > 0
                   and (indices_to_augment is None or idx in indices_to_augment)
            ]

            if not class_samples:
                print(f"⚠️ No training samples found for class '{class_name}' in dataset.")
                continue

            required_count = int(((target_ratio / 100) * original_size) - (current_ratio / 100 * original_size))
            augmented = []

            for i in range(required_count):
                img, mask, original_idx = class_samples[i % len(class_samples)]
                aug_img, aug_mask = img.clone(), mask.clone()

                for aug_func in augmentation_functions:
                    aug_img, aug_mask = aug_func(aug_img, aug_mask)

                new_images.append(aug_img)
                new_masks.append(aug_mask)
                dataset.augmented_from_idx.append(original_idx)  # <-- Track original index

                if fold_assignments:
                    new_idx = len(new_images) - 1
                    if original_idx in fold_assignments:
                        new_fold_assignments[new_idx] = fold_assignments[original_idx]
                    else:
                        print(f"Warning: original_idx {original_idx} not in fold_assignments. Assigning to fold -1.")
                        new_fold_assignments[new_idx] = -1

                augmented.append((aug_img, aug_mask))

            augmented_counts[class_name] = len(augmented)

    dataset.images = new_images
    dataset.masks = new_masks

    if fold_assignments:
        fold_assignments.update(new_fold_assignments)

    print(f"✅ Augmented counts: {augmented_counts}")
    return augmented_counts


# === Pixel-level class augmentation ===
def augment_minority_classes_pixel_level(dataset, class_labels, target_pixel_ratios, fold_assignments=None, augmentation_functions=None, max_aug_per_class=200):
    if not hasattr(dataset, "images") or not hasattr(dataset, "masks"):
        raise ValueError("Dataset must be loaded into memory with .images and .masks attributes.")

    if augmentation_functions is None:
        augmentation_functions = [apply_combined_augmentations]

    augmented_counts = {}
    new_images = list(dataset.images)
    new_masks = list(dataset.masks)
    new_fold_assignments = dict(fold_assignments) if fold_assignments else {}

    total_pixels = sum(mask.numel() for mask in dataset.masks)
    pixel_counts = {cls: 0 for cls in class_labels.values()}
    for mask in dataset.masks:
        for idx, name in class_labels.items():
            pixel_counts[name] += (mask == idx).sum().item()

    for idx, name in class_labels.items():
        current = pixel_counts[name]
        target = int(target_pixel_ratios[name] * total_pixels)
        if current >= target:
            continue

        print(f"⬆️ Pixel-level augmenting class '{name}' (current: {current}, target: {target})")
        samples = [(img, msk, i) for i, (img, msk) in enumerate(zip(dataset.images, dataset.masks)) if (msk == idx).sum().item() > 0]
        samples.sort(key=lambda x: (x[1] == idx).sum().item(), reverse=True)

        if not samples:
            print(f"⚠️ No samples found for class '{name}'.")
            continue

        i, added_pixels, augmented = 0, 0, []
        while current + added_pixels < target and len(augmented) < max_aug_per_class:
            img, msk, orig = samples[i % len(samples)]
            aug_img, aug_msk = img.clone(), msk.clone()
            for aug_func in augmentation_functions:
                aug_img, aug_msk = aug_func(aug_img, aug_msk)
            new_images.append(aug_img)
            new_masks.append(aug_msk)
            new_idx = len(new_images) - 1
            if fold_assignments:
                new_fold_assignments[new_idx] = fold_assignments.get(orig, -1)
            added_pixels += (aug_msk == idx).sum().item()
            augmented.append((aug_img, aug_msk))
            i += 1

        augmented_counts[name] = len(augmented)

    dataset.images = new_images
    dataset.masks = new_masks
    if fold_assignments:
        fold_assignments.update(new_fold_assignments)

    print(f"\n✅ Pixel-level augmented counts: {augmented_counts}")

    # Final class pixel ratios
    total_aug_pixels = sum((m == cls_idx).sum().item() for m in new_masks for cls_idx in class_labels)
    print("\n📊 Final pixel distribution:")
    for cls_idx, name in class_labels.items():
        p_count = sum((m == cls_idx).sum().item() for m in new_masks)
        print(f"  {name}: {p_count / total_aug_pixels:.2%}")

    return augmented_counts


