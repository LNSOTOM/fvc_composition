import torch
import numpy as np
from PIL import Image
import rasterio
import os
import re



def prep_mask(mask_name, replace_value=-1):
    """Read the mask, convert to grayscale, replace NaN values, and return as a numpy array along with its profile."""
    try:
        with rasterio.open(mask_name) as src:
            mask = src.read(1)  # Reading the first band
            profile = src.profile  # Get the profile containing metadata like CRS and affine transform
    except Exception as e:
        raise IOError(f"Error reading mask file {mask_name}: {e}")

    # Replace NaNs or other designated placeholder values with the specified value
    mask = np.where(np.isnan(mask), replace_value, mask)

    return mask, profile

# def prep_mask(mask_name, replace_value=-1, region_name=None):
#     """
#     Read the mask, replace NaN values, and return as a numpy array along with its profile.
#     Attempts to load the mask using the original name and falls back to a region-based naming convention if necessary.
#     """
#     mask = None
#     profile = None

#     # Attempt to load the mask file as provided
#     try:
#         print(f"Attempting to load mask: {mask_name}")
#         with rasterio.open(mask_name) as src:
#             mask = src.read(1).astype(np.float32)  # Read the first band as float
#             profile = src.profile  # Get profile containing metadata
#         print(f"Successfully loaded mask: {mask_name}")
#         return np.where(np.isnan(mask), replace_value, mask), profile
#     except rasterio.errors.RasterioIOError:
#         print(f"Failed to load mask with original name: {mask_name}")

#     # Attempt region-based naming convention if region_name is provided
#     if region_name:
#         base_name = os.path.basename(mask_name)
#         mask_name_with_region = os.path.join(os.path.dirname(mask_name), f"{region_name}_{base_name}")

#         # Attempt to load with the region-based naming convention
#         try:
#             print(f"Attempting region-prefixed load for: {mask_name_with_region}")
#             with rasterio.open(mask_name_with_region) as src:
#                 mask = src.read(1).astype(np.float32)
#                 profile = src.profile
#             print(f"Successfully loaded mask with region prefix: {mask_name_with_region}")
#             return np.where(np.isnan(mask), replace_value, mask), profile
#         except rasterio.errors.RasterioIOError as e:
#             print(f"Failed to load mask with region prefix: {mask_name_with_region} - {e}")

#     # If both attempts fail, raise an error
#     raise FileNotFoundError(
#         f"Mask file not found with either original name: {mask_name} or region-prefixed name: {mask_name_with_region if region_name else 'N/A'}"
#     )


def prep_mask_preserve_nan(mask_name):
    """
    Read the mask and return it as a numpy array along with its profile, preserving NaN values.
    
    Args:
        mask_name (str): The file path to the mask.

    Returns:
        np.ndarray: The mask with NaN values preserved.
        dict: The profile of the mask containing metadata like CRS and affine transform.
    """
    try:
        with rasterio.open(mask_name) as src:
            mask = src.read(1)  # Reading the first band
            profile = src.profile  # Get the profile containing metadata like CRS and affine transform
    except Exception as e:
        raise IOError(f"Error reading mask file {mask_name}: {e}")

    # Directly return the mask without modifying NaN values
    return mask, profile


# def convertMask_to_tensor(data, dtype=torch.long):
#     """ Convert numpy array to a PyTorch tensor of specified type. """
#     return torch.from_numpy(data).type(dtype)

def convertMask_to_tensor(data, dtype=torch.long):
    """ Convert numpy array to a PyTorch tensor of specified type. """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(dtype)
    elif isinstance(data, torch.Tensor):
        return data.type(dtype)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, but got {type(data)}")