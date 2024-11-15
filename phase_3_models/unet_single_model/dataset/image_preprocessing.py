import os
import rasterio
import numpy as np
import torch
import re


# Function to load raw multispectral data
def load_raw_multispectral_image(img_name):
    """ Load raw multispectral data using rasterio without normalization or contrast stretching. """
    # Check if the file exists
    if not os.path.exists(img_name):
        raise FileNotFoundError(f"File does not exist: {img_name}")

    try:
        with rasterio.open(img_name) as src:
            image = src.read().astype(np.float32)
            profile = src.profile  # Extract the profile containing metadata
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening file with rasterio: {img_name}")
        raise e

    return image, profile



# def load_raw_multispectral_image(img_name, region_name=None):
#     """
#     Load raw multispectral data using rasterio without normalization or contrast stretching.
#     Attempts to load the file with the original name and then with a region-based prefix if specified.
#     """
#     image = None
#     profile = None

#     # Attempt to load the image file as-is
#     try:
#         print(f"Attempting to load image with original path: {img_name}")
#         with rasterio.open(img_name) as src:
#             image = src.read().astype(np.float32)
#             profile = src.profile
#         print(f"Successfully loaded image: {img_name}")
#         return image, profile
#     except rasterio.errors.RasterioIOError:
#         print(f"Failed to load image with original name: {img_name}")

#     # If loading with the original name fails, try with region prefix if provided
#     if region_name:
#         # Construct the new filename with region prefix
#         base_name = os.path.basename(img_name)
#         img_name_with_region = os.path.join(os.path.dirname(img_name), f"{region_name}_{base_name}")
        
#         # Attempt to load with the region-based naming convention
#         try:
#             print(f"Attempting region-prefixed load for: {img_name_with_region}")
#             with rasterio.open(img_name_with_region) as src:
#                 image = src.read().astype(np.float32)
#                 profile = src.profile
#             print(f"Successfully loaded image with region prefix: {img_name_with_region}")
#             return image, profile
#         except rasterio.errors.RasterioIOError as e:
#             print(f"Failed to load image with region prefix: {img_name_with_region} - {e}")

#     # If both attempts fail, raise an error
#     raise FileNotFoundError(
#         f"Image file not found with either original name: {img_name} or region-prefixed name: {img_name_with_region if region_name else 'N/A'}"
#     )


''' Normalisation based on local statistics (drawn from the pixels in each individual tile)
    CNN function (or how they transform image data into information); they primarily work on patterns 
    and relative pixel values rather than absolute values. 
    This makes them also quite robust against differences in illumination conditions.
'''
# a) Normalise image to [0, 1]
def prep_normalise_image(img_name):
    """ Normalise each channel of the image to [0, 1] by dividing by the maximum value in each channel. """
    with rasterio.open(img_name) as src:
        image = src.read().astype(np.float32)  # Read the raster data as float32
        normalise_image = np.zeros_like(image)  # Initialize a zero array with the same shape as the input image
        for i in range(image.shape[0]):  # Iterate over each channel
            max_val = image[i].max()
            if max_val > 0:  # Avoid division by zero
                normalise_image[i] = image[i] / max_val
                
        # normalise_image = np.round(normalise_image, 4)  # Round the normalized image to 4 decimal places
        profile = src.profile  # Extract the profile containing metadata
        return normalise_image, profile  # Return the normalized image and the profile


# b) Calculate 99th percentile and normalize image to [0, 1]
'''implement the percentile-based normalization (or contrast stretching)'''
''' represent 99% of the maximum value instead of the absolute maximum value, 
    you need to calculate the 99th percentile of the image data and normalize based on that value.'''
'''using the 1st and 99th percentiles for normalization, you're essentially stretching the pixel values 
    between these two percentiles to cover the full range of [0, 1].referred to as contrast stretching, 
    which can enhance the contrast in an image.
'''
def prep_contrast_stretch_image(img_name):
    """ Apply contrast stretching to each channel of the image using the 1st and 99th percentiles. """
    with rasterio.open(img_name) as src:
        image = src.read().astype(np.float32)  # Read the raster data as float32
        contrast_stretched_image = np.zeros_like(image, dtype=np.float32)  # Initialize a zero array

        for i in range(image.shape[0]):  # Iterate over each channel
            channel = image[i]
            p1, p99 = np.percentile(channel, [1, 99])
            if p99 > p1:  # Avoid division by zero or negative values
                contrast_stretched_image[i] = (channel - p1) / (p99 - p1)
                contrast_stretched_image[i] = np.clip(contrast_stretched_image[i], 0, 1)  # Clip values to [0, 1]
            else:
                contrast_stretched_image[i] = np.zeros_like(channel)  # Handle cases where no adjustment is needed
        
        
        # contrast_stretched_image = np.round(contrast_stretched_image, 4)  # Round the contrast-stretched image to 4 decimal places
        profile = src.profile  # Extract the profile containing metadata
        return contrast_stretched_image, profile  # Return the contrast-stretched image and the profile


def convertImg_to_tensor(data, dtype=torch.float32):
    """ Convert numpy array to a PyTorch tensor of specified type. """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(dtype)
    elif isinstance(data, torch.Tensor):
        return data.type(dtype)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, but got {type(data)}")


########################################################
############# RGB imagery
'''function for RGB 8-bit images'''
def load_raw_rgb_image(img_name):
    """ Load raw RGB 8-bit data using rasterio without normalization or contrast stretching. """
    with rasterio.open(img_name) as src:
        image = src.read().astype(np.uint8)  # Read the raster data as 8-bit unsigned integers (0-255)
        profile = src.profile  # Extract the profile containing metadata

        # Iterate over each channel for additional processing and statistics
        for i in range(image.shape[0]):  # Assuming the first dimension is the channel (RGB)
            channel = image[i]

            # # Calculate statistics for the channel
            # channel_mean = channel.mean()
            # channel_std = channel.std()
            # channel_min = channel.min()
            # channel_max = channel.max()

            # # Print or log the statistics
            # print(f"Channel {i+1} statistics (RGB):")
            # print(f"  Mean: {channel_mean}")
            # print(f"  Standard Deviation: {channel_std}")
            # print(f"  Min: {channel_min}")
            # print(f"  Max: {channel_max}")

            # Example: No transformation needed for raw RGB data, just stats
            # You can add other operations here if needed, like contrast adjustment

    return image, profile  # Return the raw RGB image and the profile


def convertImg_to_tensor(data, dtype=torch.uint8):
    """ Convert numpy array to a PyTorch tensor of specified type, adapted for 8-bit RGB images. """
    if isinstance(data, np.ndarray):
        # Ensure the data is converted to uint8 if it's not already
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)
        return torch.from_numpy(data).type(dtype)
    elif isinstance(data, torch.Tensor):
        # If it's already a tensor, convert to the specified dtype (likely uint8 for 8-bit images)
        return data.type(dtype)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, but got {type(data)}")
