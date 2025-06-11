import torch
import numpy as np
from PIL import Image
import rasterio
import cv2


#     return mask - orginal
# def prep_mask(mask_name, replace_value=-1):
#     """Read the mask, convert to grayscale, replace NaN values, and return as a numpy array along with its profile."""
#     try:
#         with rasterio.open(mask_name) as src:
#             mask = src.read(1)  # Reading the first band
#             profile = src.profile  # Get the profile containing metadata like CRS and affine transform
#          # Resize mask to match image dimensions if needed
#         if 'height' in profile and 'width' in profile:
#             target_height, target_width = profile['height'], profile['width']
#             if mask.shape != (target_height, target_width):
#                 print(f"Resizing mask from {mask.shape} to {(target_height, target_width)}")
#                 mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
#     except Exception as e:
#         raise IOError(f"Error reading mask file {mask_name}: {e}")

#     # Replace NaNs or other designated placeholder values with the specified value
#     mask = np.where(np.isnan(mask), replace_value, mask)

#     return mask, profile

###new to handle n/a and nan values
def prep_mask(mask_name, replace_value=-1):
    """
    Read a raster mask, replace NaN and NoData values with a given placeholder (e.g., -1),
    and return the cleaned mask along with its profile.
    """
    try:
        with rasterio.open(mask_name) as src:
            mask = src.read(1).astype(np.float32)  # Force float32 to allow NaNs
            profile = src.profile
            nodata_val = src.nodata  # This will be None if 'No-Data' is 'n/a'

        # Resize mask if needed
        if 'height' in profile and 'width' in profile:
            target_height, target_width = profile['height'], profile['width']
            if mask.shape != (target_height, target_width):
                print(f"Resizing mask from {mask.shape} to {(target_height, target_width)}")
                mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    except Exception as e:
        raise IOError(f"Error reading mask file {mask_name}: {e}")

    # Replace NaNs and nodata values with the specified replace_value (e.g., -1)
    if nodata_val is not None and not np.isnan(nodata_val):
        mask = np.where((mask == nodata_val) | np.isnan(mask), replace_value, mask)
    else:
        mask = np.where(np.isnan(mask), replace_value, mask)

    return mask, profile


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


