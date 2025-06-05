import numpy as np
import torch
import torchvision.transforms as transforms
from collections import Counter
import random
from copy import deepcopy

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
    transform = transforms.RandomAffine(
        degrees=20,
        translate=(0.15, 0.15),
        shear=(0, 0.2)
    )
    channels = []
    for i in range(image.shape[0]):
        channel = transforms.ToPILImage()(image[i].unsqueeze(0))
        channel = transform(channel)
        channels.append(transforms.ToTensor()(channel).squeeze(0))

    # Transform mask and ensure it's still an integer class label
    mask = transforms.ToPILImage()(mask.to(torch.uint8))  # PIL expects uint8
    mask = transform(mask)
    mask = transforms.ToTensor()(mask).squeeze(0)

    mask = torch.round(mask).long()  # ⬅️ force back to integer class labels

    return torch.stack(channels), mask


# Function to combine all augmentations
def apply_combined_augmentations(image, mask):
    """
    Apply a combination of augmentations: color jitter, affine transformations, and flips.
    """
    # Apply color jitter
    image, mask = apply_color_jitter(image, mask)

    # Apply random affine transformations (shifting and shearing included)
    image, mask = apply_random_affine(image, mask)

    # Randomly apply vertical or horizontal flip
    if torch.rand(1).item() > 0.5:
        image, mask = apply_vertical_flip(image, mask)
    if torch.rand(1).item() > 0.5:
        image, mask = apply_horizontal_flip(image, mask)

    return image, mask

def augment_minority_class(dataset, class_label, augmentation_functions, target_ratio=0.2, fold_assignments=None):
    """
    Augment the dataset for the minority class using apply_combined_augmentations.

    Args:
        dataset: Dataset object.
        class_label (int): Class index to augment.
        augmentation_functions (list): Unused now, replaced by apply_combined_augmentations.
        target_ratio (float): Target ratio (ignored in current implementation).
        fold_assignments (dict, optional): Mapping from dataset index to fold ID.

    Returns:
        list: A list of tuples (augmented_index, original_index).
    """
    augmented_samples = []
    for i in range(len(dataset)):
        image, mask = dataset[i]
        if (mask == class_label).any():
            aug_image, aug_mask = apply_combined_augmentations(image, mask)
            dataset.images.append(aug_image)
            dataset.masks.append(aug_mask)
            aug_idx = len(dataset.images) - 1
            augmented_samples.append((aug_idx, i))

    print(f"✅ Augmented {len(augmented_samples)} samples for class {class_label}.")

    # Assign augmented samples to same fold as originals
    if fold_assignments is not None:
        for aug_idx, orig_idx in augmented_samples:
            fold = fold_assignments[orig_idx]
            fold_assignments[aug_idx] = fold

        # Report fold distribution
        fold_counts = Counter([fold_assignments[aug_idx] for aug_idx, _ in augmented_samples])
        print("\n🧩 Distribution of augmented samples across folds:")
        for fold_id in range(5):
            print(f"  Fold {fold_id}: {fold_counts.get(fold_id, 0)} samples")

    return augmented_samples


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
#         target_ratio = target_ratios.get(class_name, 0.1)  # Default target ratio is 10%
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

def augment_minority_classes(dataset, class_distributions, class_labels, target_ratios, fold_assignments=None, augmentation_functions=None):
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

            class_samples = [
                (img, mask, idx)
                for idx, (img, mask) in enumerate(zip(dataset.images, dataset.masks))
                if (mask == class_idx).sum().item() > 0
            ]

            if not class_samples:
                print(f"⚠️ No samples found for class '{class_name}' in dataset.")
                continue

            required_count = int(((target_ratio / 100) * original_size) - (current_ratio / 100 * original_size))
            augmented = []

            for i in range(required_count):
                img, mask, original_idx = class_samples[i % len(class_samples)]
                aug_img, aug_mask = img.clone(), mask.clone()

                for aug_func in augmentation_functions:
                    aug_img, aug_mask = aug_func((aug_img, aug_mask))

                new_images.append(aug_img)
                new_masks.append(aug_mask)

                if fold_assignments:
                    new_idx = len(new_images) - 1
                    if original_idx in fold_assignments:
                        new_fold_assignments[new_idx] = fold_assignments[original_idx]
                    else:
                        # Optionally, assign to a default fold or print a warning
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
