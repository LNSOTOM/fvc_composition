import numpy as np
import torch
import torchvision.transforms as transforms

# Function to apply percentile filter and stretch
# def apply_percentile_filter_and_stretch(img, lower_percentile=2, upper_percentile=98):
#     img_stretched = np.zeros_like(img, dtype=np.float32)
#     for i in range(img.shape[0]):
#         lower = np.percentile(img[i, :, :], lower_percentile)
#         upper = np.percentile(img[i, :, :], upper_percentile)
#         img_stretched[i, :, :] = np.clip((img[i, :, :] - lower) / (upper - lower), 0, 1)
#     return img_stretched

# Function to apply ColorJitter transformation
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
