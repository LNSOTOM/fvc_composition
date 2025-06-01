import numpy as np
import torch
import torchvision.transforms as transforms

# Function to apply ColorJitter transformation (brightness and contrast adjustments)
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

# Function to apply random affine transformation (shifting and shearing included)
def apply_random_affine(image, mask):
    transform = transforms.RandomAffine(
        degrees=20,  # Random rotation up to ±20 degrees
        translate=(0.15, 0.15),  # Random shifting up to 15% of image size
        shear=(0, 0.2)  # Random shearing up to 0.2 radians
    )
    channels = []
    for i in range(image.shape[0]):
        channel = transforms.ToPILImage()(image[i].unsqueeze(0))
        channel = transform(channel)
        channels.append(transforms.ToTensor()(channel).squeeze(0))

    # Convert mask to float32, apply transformation, and convert back to float32
    mask = mask.to(torch.float32)  # Ensure mask is in float32
    mask = transforms.ToPILImage()(mask)
    mask = transform(mask)
    mask = transforms.ToTensor()(mask).to(torch.float32)  # Keep mask in float32

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
