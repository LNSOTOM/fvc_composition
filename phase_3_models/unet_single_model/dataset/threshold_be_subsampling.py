import numpy as np
import random


'''
Subsampling Function to dataset by removing tiles with a high soil cover.  
This involves first calculating the soil cover in each tile, then removing a random subset of tiles 
where the soil cover exceeds a certain threshold.
'''

# 1. Calculate Soil Cover Percentage in Each Tile:
'''
The soil cover percentage in each tile. This can be done by counting the number of soil pixels 
and dividing by the total number of pixels in the tile.
'''
# def calculate_soil_cover(mask, soil_class=0):
#     # Ensure mask is a NumPy array of type float32
#     if not isinstance(mask, np.ndarray):
#         mask = np.array(mask, dtype=np.float32)
    
#     total_pixels = mask.size
#     soil_pixels = np.sum(mask == soil_class)
#     soil_cover_percentage = (soil_pixels / total_pixels) * 100
#     return soil_cover_percentage

def calculate_soil_cover(mask, soil_class=0):
    """Calculate the percentage of soil cover in a mask, ignoring NaN values."""
    # Ensure mask is a NumPy array of type float32
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=np.float32)
    
    # Ignore NaN values by using np.isnan
    valid_mask = mask[~np.isnan(mask)]
    total_pixels = valid_mask.size
    
    # Handle case where no valid pixels are present
    if total_pixels == 0:
        return 0  # Or another sentinel value or raise an exception, depending on your needs
    
    soil_pixels = np.sum(valid_mask == soil_class)
    soil_cover_percentage = (soil_pixels / total_pixels) * 100
    return soil_cover_percentage


# def calculate_soil_cover(mask, soil_class=0):
#     """Calculate the percentage of soil cover in a mask, ignoring NaN values."""
#     # Ignore NaN values by using np.isnan
#     valid_mask = mask[~np.isnan(mask)]
#     total_pixels = valid_mask.size
    
#     # Handle case where no valid pixels are present
#     if total_pixels == 0:
#         return 0  # Or another sentinel value or raise an exception, depending on your needs
    
#     soil_pixels = np.sum(valid_mask == soil_class)
#     soil_cover_percentage = (soil_pixels / total_pixels) * 100
#     return soil_cover_percentage

# 2. Remove Tiles Based on Soil Cover Threshold:
'''
Subsample the dataset by removing tiles with high soil cover.
'''
def subsample_tiles(images, masks, soil_threshold, soil_class=0, removal_ratio=0.5):
    assert len(images) == len(masks), "Images and masks lists must have the same length."
    
    remaining_images = []
    remaining_masks = []
    subsampled_indices = []  # To keep track of the indices of the selected images and masks
    
    for idx, (img, mask) in enumerate(zip(images, masks)):
        soil_cover = calculate_soil_cover(mask, soil_class)
        print(f"Processing tile {idx}: soil cover = {soil_cover}%")  # Debug statement
        
        if soil_cover <= soil_threshold:
            print(f"Keeping tile {idx} (soil cover {soil_cover}% <= threshold {soil_threshold}%)")  # Debug statement
            remaining_images.append(img)
            remaining_masks.append(mask)
            subsampled_indices.append(idx)
        else:
            if random.random() > removal_ratio:
                print(f"Randomly keeping tile {idx} despite high soil cover {soil_cover}% > threshold {soil_threshold}%")  # Debug statement
                remaining_images.append(img)
                remaining_masks.append(mask)
                subsampled_indices.append(idx)
            else:
                print(f"Removing tile {idx} due to high soil cover {soil_cover}% > threshold {soil_threshold}%")  # Debug statement
    
    print(f"Total tiles kept after subsampling: {len(remaining_images)}")  # Debug statement
    return remaining_images, remaining_masks, subsampled_indices

def estimate_class_frequencies(masks, num_classes):
    """
    Estimate the frequency of each class in the dataset.
    
    Args:
        masks (list of np.array): List of mask arrays.
        num_classes (int): Number of classes.
        
    Returns:
        np.array: Array of class frequencies.
    """
    class_counts = np.zeros(num_classes, dtype=int)
    
    for mask in masks:
        # Ignore NaN values by using np.isnan
        valid_mask = mask[~np.isnan(mask)]
        
        # Ensure valid_mask is a NumPy array of type float32
        if not isinstance(valid_mask, np.ndarray):
            valid_mask = np.array(valid_mask, dtype=np.float32)
        
        unique, counts = np.unique(valid_mask, return_counts=True)
        for cls, count in zip(unique, counts):
            # Ensure cls is an integer and within the valid range
            cls = int(cls)
            if 0 <= cls < num_classes:
                class_counts[cls] += count
    
    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    return class_frequencies

#####################
#######faster
# import numpy as np
# import dask.array as da
# import random

# def calculate_soil_cover(mask, soil_class=0):
#     # Ensure mask is a NumPy array of type float32
#     if not isinstance(mask, np.ndarray):
#         mask = np.array(mask, dtype=np.float32)
    
#     total_pixels = mask.size
#     soil_pixels = np.sum(mask == soil_class)
#     soil_cover_percentage = (soil_pixels / total_pixels) * 100
#     return soil_cover_percentage

# def subsample_tiles(images, masks, soil_threshold, soil_class=0, removal_ratio=0.5):
#     assert len(images) == len(masks), "Images and masks lists must have the same length."
    
#     remaining_images = []
#     remaining_masks = []
    
#     for img, mask in zip(images, masks):
#         soil_cover = calculate_soil_cover(mask, soil_class)
#         if soil_cover <= soil_threshold:
#             remaining_images.append(img)
#             remaining_masks.append(mask)
#         else:
#             if random.random() > removal_ratio:
#                 remaining_images.append(img)
#                 remaining_masks.append(mask)
    
#     return da.from_array(remaining_images), da.from_array(remaining_masks)

# def estimate_class_frequencies(masks, num_classes):
#     """
#     Estimate the frequency of each class in the dataset.
    
#     Args:
#         masks (list of np.array): List of mask arrays.
#         num_classes (int): Number of classes.
        
#     Returns:
#         np.array: Array of class frequencies.
#     """
#     class_counts = np.zeros(num_classes, dtype=int)
    
#     for mask in masks:
#         # Ensure mask is a NumPy array of type float32
#         if not isinstance(mask, np.ndarray):
#             mask = np.array(mask, dtype=np.float32)
        
#         unique, counts = np.unique(mask, return_counts=True)
#         for cls, count in zip(unique, counts):
#             # Ensure cls is an integer and within the valid range
#             cls = int(cls)
#             if 0 <= cls < num_classes:
#                 class_counts[cls] += count
    
#     total_pixels = np.sum(class_counts)
#     class_frequencies = class_counts / total_pixels
#     return class_frequencies
