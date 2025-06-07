import numpy as np
import torch
import torchvision.transforms as transforms
from collections import Counter
import random
from copy import deepcopy

import cv2
import numpy as np

# Function to apply ColorJitter transformation (brightness and contrast adjustments)
def transform_image_by_channels(image, transform_fn):
    """
    Apply a transform function (e.g., PIL-based) to each channel of a tensor image.

    Args:
        image (Tensor): Tensor of shape [C, H, W]
        transform_fn (function): Transform to apply on each channel as PIL Image

    Returns:
        Tensor: Transformed image with same shape
    """
    transformed_channels = []
    for c in image:
        pil_img = transforms.ToPILImage()(c.unsqueeze(0))
        transformed = transform_fn(pil_img)
        transformed_channels.append(transforms.ToTensor()(transformed).squeeze(0))
    return torch.stack(transformed_channels)


def apply_color_jitter(image, mask):
    """
    Apply color jitter to the image (not the mask).

    Args:
        image (Tensor): [C, H, W]
        mask (Tensor): [H, W]

    Returns:
        Tuple[Tensor, Tensor]: (augmented image, unchanged mask)
    """
    transform = transforms.ColorJitter(
        brightness=(0.9, 1.1),
        contrast=(0.8, 1.2)
    )
    image = transform_image_by_channels(image, transform)
    return image, mask

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

# Function to apply random affine transformation (shifting and shearing included)
def apply_random_affine(image, mask):
    """
    Apply random affine transformation while preserving exact mask values.
    """
    # First, store all unique class values in the original mask
    original_unique = torch.unique(mask).tolist()
    original_dtype = mask.dtype
    
    # Create the affine transformation parameters
    angle = random.uniform(-20, 20)
    translate = [random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)]
    scale = random.uniform(0.9, 1.1)
    shear = random.uniform(0, 0.2)
    
    # Process the image using bilinear interpolation
    channels = []
    for i in range(image.shape[0]):
        # Convert channel to PIL
        channel_pil = transforms.ToPILImage()(image[i].unsqueeze(0))
        
        # Apply affine transform with BILINEAR interpolation (for image)
        transformed_channel = transforms.functional.affine(
            channel_pil,
            angle=angle, 
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        
        # Convert back to tensor
        channel_tensor = transforms.ToTensor()(transformed_channel).squeeze(0)
        channels.append(channel_tensor)
    
    # Stack the transformed channels
    transformed_image = torch.stack(channels)
    
    # For the mask, we need to handle each class separately to preserve exact values
    h, w = mask.shape
    transformed_mask = torch.ones((h, w), dtype=original_dtype) * -1  # Start with -1 (background)
    
    # Process each class value separately
    for class_val in original_unique:
        # Create a binary mask for this class
        binary_mask = (mask == class_val).to(torch.uint8) * 255
        
        # Convert to PIL
        binary_mask_pil = transforms.ToPILImage()(binary_mask)
        
        # Apply the SAME affine transform with NEAREST interpolation
        transformed_binary = transforms.functional.affine(
            binary_mask_pil,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=transforms.InterpolationMode.NEAREST
        )
        
        # Convert back to tensor and threshold to get binary mask
        transformed_binary_tensor = transforms.ToTensor()(transformed_binary).squeeze(0)
        
        # This is the key fix: ensure binary_result is boolean, not Long
        binary_result = transformed_binary_tensor > 0.5  # Creates a boolean tensor
        
        # Convert class_val to scalar if it's a tensor
        class_val_scalar = class_val if not isinstance(class_val, torch.Tensor) else class_val.item()
        
        # Apply where with boolean condition
        transformed_mask = torch.where(binary_result, 
                                      torch.tensor(class_val_scalar, dtype=original_dtype), 
                                      transformed_mask)
    
    # Verify class preservation
    new_unique = torch.unique(transformed_mask).tolist()
    preserved = all(val in new_unique for val in original_unique if val >= 0)
    
    if not preserved:
        missing = set(val for val in original_unique if val >= 0) - set(val for val in new_unique if val >= 0)
        print(f"WARNING: Lost classes during affine augmentation! Missing: {missing}")
    
    return transformed_image, transformed_mask


# Function to combine all augmentations
def apply_combined_augmentations(image, mask):
    """Apply a combination of augmentations to both image and mask."""
    original_unique = torch.unique(mask)
    # print("Before augmentation - Unique classes in mask:", original_unique)
    
    # Store the original mask dtype to restore later
    original_dtype = mask.dtype
    
    # 1. Random horizontal flip (50% chance)
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-1])
        mask = torch.flip(mask, dims=[-1])
    
    # 2. Random vertical flip (50% chance)
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-2])
        mask = torch.flip(mask, dims=[-2])
    
    # 3. Apply color jitter - image only
    if random.random() > 0.5:
        for c in range(image.shape[0]):
            image[c] = image[c] * (0.8 + 0.4 * random.random())
    
    # 4. Random rotation - be very careful with mask interpolation
    if random.random() > 0.3:  # 70% chance of rotation
        angle = random.uniform(-10, 10)
        # Apply rotation to image using bilinear interpolation
        image = transforms.functional.rotate(
            image, angle, interpolation=transforms.InterpolationMode.NEAREST
        )
        # Apply rotation to mask using NEAREST interpolation to preserve class values
        mask = transforms.functional.rotate(
            mask.unsqueeze(0), angle, interpolation=transforms.InterpolationMode.NEAREST
        ).squeeze(0)
      
    # Ensure mask has the right dtype and values
    mask = mask.to(original_dtype)
    
    # Verify classes are preserved
    # new_unique = torch.unique(mask)
    # print("After augmentation - Unique classes in mask:", new_unique)
    
    # Check if we've lost any non-negative classes
    # positive_classes_before = set([v.item() for v in original_unique if v >= 0])
    # positive_classes_after = set([v.item() for v in new_unique if v >= 0])
    # if not positive_classes_after.issuperset(positive_classes_before):
    #     missing = positive_classes_before - positive_classes_after
    #     print(f"WARNING: Lost positive classes during augmentation: {missing}")
    
    return image, mask

# def apply_combined_augmentations(image, mask):
#     """Apply a combination of augmentations to both image and mask.
    
#     This improved implementation uses modular augmentation functions
#     and tracks applied transformations for better monitoring.
    
#     Args:
#         image (Tensor): Image tensor of shape [C, H, W]
#         mask (Tensor): Mask tensor of shape [H, W]
        
#     Returns:
#         tuple: (augmented_image, augmented_mask)
#     """
#     # Store original properties for verification
#     original_unique = torch.unique(mask).tolist()
#     original_dtype = mask.dtype
#     applied = []
    
#     # 1. Random horizontal flip (50% chance)
#     if random.random() > 0.5:
#         image, mask = apply_horizontal_flip(image, mask)
#         applied.append("h_flip")
    
#     # 2. Random vertical flip (50% chance)
#     if random.random() > 0.5:
#         image, mask = apply_vertical_flip(image, mask)
#         applied.append("v_flip")
    
#     # 3. Apply color jitter to image only (50% chance)
#     if random.random() > 0.5:
#         image, mask = apply_color_jitter(image, mask)
#         applied.append("color_jitter")
    
#     # 4. Random rotation (30% chance)
#     if random.random() > 0.7:
#         angle = random.uniform(-10, 10)
#         image = transforms.functional.rotate(
#             image, angle, interpolation=transforms.InterpolationMode.BILINEAR
#         )
#         mask = transforms.functional.rotate(
#             mask.unsqueeze(0), angle, interpolation=transforms.InterpolationMode.NEAREST
#         ).squeeze(0)
#         applied.append("rotation")
    
#     # 5. Apply random affine with smaller probability (20% chance)
#     if random.random() > 0.8:
#         image, mask = apply_random_affine(image, mask)
#         applied.append("affine")
    
#     # Ensure mask has the right dtype
#     mask = mask.to(original_dtype)
    
#     # Verify class preservation
#     new_unique = torch.unique(mask).tolist()
#     preserved = all(val in new_unique for val in original_unique if val >= 0)
    
#     if not preserved:
#         missing = {val for val in original_unique if val >= 0 and val not in new_unique}
#         print(f"WARNING: Lost classes {missing} after applying {applied}")
    
#     return image, mask

####################
# def augment_minority_class(dataset, class_label, augmentation_functions, target_ratio=0.2, fold_assignments=None):
#     """
#     Augment the dataset for the minority class using apply_combined_augmentations.

#     Args:
#         dataset: Dataset object.
#         class_label (int): Class index to augment.
#         augmentation_functions (list): Unused now, replaced by apply_combined_augmentations.
#         target_ratio (float): Target ratio (ignored in current implementation).
#         fold_assignments (dict, optional): Mapping from dataset index to fold ID.

#     Returns:
#         list: A list of tuples (augmented_index, original_index).
#     """
#     augmented_samples = []
#     for i in range(len(dataset)):
#         image, mask = dataset[i]
#         if (mask == class_label).any():
#             aug_image, aug_mask = apply_combined_augmentations(image, mask)
#             dataset.images.append(aug_image)
#             dataset.masks.append(aug_mask)
#             dataset.augmented_from_idx.append(original_idx)  # <-- Add this line
#             aug_idx = len(dataset.images) - 1
#             augmented_samples.append((aug_idx, i))

#     print(f"✅ Augmented {len(augmented_samples)} samples for class {class_label}.")

#     # Assign augmented samples to same fold as originals
#     if fold_assignments is not None:
#         for aug_idx, orig_idx in augmented_samples:
#             fold = fold_assignments[orig_idx]
#             fold_assignments[aug_idx] = fold

#         # Report fold distribution
#         fold_counts = Counter([fold_assignments[aug_idx] for aug_idx, _ in augmented_samples])
#         print("\n🧩 Distribution of augmented samples across folds:")
#         for fold_id in range(5):
#             print(f"  Fold {fold_id}: {fold_counts.get(fold_id, 0)} samples")

#     return augmented_samples


# def augment_minority_classes(dataset, class_distributions, class_labels, target_ratios, fold_assignments):
#     """
#     Augment the dataset for classes below their target ratios, ensuring augmented samples remain in the same fold.

#     Args:
#         dataset: Dataset object.
#         class_distributions: Dict of current class percentages.
#         class_labels: Dict mapping class index to class name.
#         target_ratios: Dict mapping class name to target ratio.
#         fold_assignments: Dict mapping sample index to fold.

#     Returns:
#         dict: Number of augmented samples per class.
#     """
#     augmented_counts = {}
#     for class_idx, class_name in class_labels.items():
#         current_ratio = class_distributions.get(class_name, 0)
#         target_ratio = target_ratios.get(class_name, 0.1) # Default target ratio is 10%
#         if current_ratio < target_ratio * 100:
#             print(f"Augmenting class '{class_name}' (current ratio: {current_ratio:.2f}%, target ratio: {target_ratio * 100:.2f}%)")
#             augmented_samples = augment_minority_class(
#                 dataset, class_label=class_idx, augmentation_functions=[], target_ratio=target_ratio
#             )
#             for sample, original_idx in augmented_samples:
#                 if fold_assignments is not None:
#                     fold = fold_assignments[original_idx]
#                     fold_assignments[sample] = fold
#             augmented_counts[class_name] = len(augmented_samples)
#     print(f"Augmented counts: {augmented_counts}")
#     return augmented_counts

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
