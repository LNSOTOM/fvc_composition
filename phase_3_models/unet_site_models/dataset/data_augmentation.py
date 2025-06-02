import numpy as np
import torch
import torchvision.transforms as transforms

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


def augment_minority_classes(dataset, class_distributions, class_labels, target_ratios, fold_assignments):
    """
    Augment the dataset for classes below their target ratios, ensuring augmented samples remain in the same fold.

    Args:
        dataset: Dataset object.
        class_distributions: Dict of current class percentages.
        class_labels: Dict mapping class index to class name.
        target_ratios: Dict mapping class name to target ratio.
        fold_assignments: Dict mapping sample index to fold.

    Returns:
        dict: Number of augmented samples per class.
    """
    augmented_counts = {}
    for class_idx, class_name in class_labels.items():
        current_ratio = class_distributions.get(class_name, 0)
        target_ratio = target_ratios.get(class_name, 0.1)  # Default target ratio is 10%
        if current_ratio < target_ratio * 100:
            print(f"Augmenting class '{class_name}' (current ratio: {current_ratio:.2f}%, target ratio: {target_ratio * 100:.2f}%)")
            augmented_samples = augment_minority_class(
                dataset, class_label=class_idx, augmentation_functions=[], target_ratio=target_ratio
            )
            for sample, original_idx in augmented_samples:
                fold = fold_assignments[original_idx]
                fold_assignments[sample] = fold
            augmented_counts[class_name] = len(augmented_samples)
    print(f"Augmented counts: {augmented_counts}")
    return augmented_counts

