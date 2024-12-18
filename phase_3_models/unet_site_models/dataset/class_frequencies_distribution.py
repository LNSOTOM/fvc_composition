
#####################################
#%%
import os
import numpy as np
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import rasterio
import time

# Original dataset
# IMAGE_FOLDER = [
#     # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/predictors_5b',  # low
#     # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/predictors_5b',  # medium
#     '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/predictors_5b',  # dense
# ]

# MASK_FOLDER = [
#     # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/mask_fvc',  # low
#     # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/mask_fvc',  # medium
#     '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc',  # dense
# ]

# Subsample dataset
IMAGE_FOLDER = [
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/predictors_5b_subsample'  #low
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/low/predictor_5b_subsample'  #sites_low
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/predictors_5b_subsample'  #medium
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/medium/predictors_5b_subsample'  #sites_medium
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/predictors_5b_subsample'  #dense
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/dense/predictor_5b_subsample'  #sites_dense
]
# os.makedirs(SUBSAMPLE_IMAGE_DIR, exist_ok=True)

MASK_FOLDER = [
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/mask_fvc_subsample'  #low
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/low/mask_fvc_subsample' #sites_low
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/mask_fvc_subsample'  #medium
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/medium/mask_fvc_subsample' #sites_medium
                    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample'
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/dense/mask_fvc_subsample' #sites_dense
]


#%%
# Logging function
def log_message(message, log_file="dataset_log.txt"):
    """
    Log a message to a specified log file.

    Args:
        message (str): Message to log.
        log_file (str): Path to the log file.
    """
    with open(log_file, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Helper function to load raster data
def load_raster(file_path, num_channels=None):
    """
    Load a raster file and return its data as a NumPy array.

    Args:
        file_path (str): Path to the raster file.
        num_channels (int, optional): Number of channels to extract.

    Returns:
        np.array: The raster data.
    """
    with rasterio.open(file_path) as src:
        if num_channels:
            data = src.read(list(range(1, num_channels + 1)))  # Read specified channels
        else:
            data = src.read(1)  # Read the first band for masks
    return data

# Dataset extraction with ProgressBar
def extract_images_and_masks(image_folder, mask_folder, num_channels=5):
    """
    Extract image and mask data from folders using Dask and a ProgressBar.

    Args:
        image_folder (str): Path to the folder containing image files.
        mask_folder (str): Path to the folder containing mask files.
        num_channels (int): Number of channels in the images.

    Returns:
        Tuple of (list of np.array, list of np.array): Images and masks.
    """
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.tif', '.tiff'))])
    mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith(('.tif', '.tiff'))])

    if len(image_files) != len(mask_files):
        raise ValueError("Mismatch in the number of images and masks.")

    # Lazy loading of data
    def get_data(idx):
        return load_raster(image_files[idx], num_channels=num_channels), load_raster(mask_files[idx])

    # Use ProgressBar for monitoring
    with ProgressBar():
        lazy_results = [delayed(get_data)(i) for i in range(len(image_files))]
        images, masks = zip(*compute(*lazy_results))

    return images, masks


#%%
# Optimized class frequency estimation
def estimate_class_frequencies(masks, num_classes):
    """
    Estimate the frequency of each class in a mask.

    Args:
        masks (list of np.array): List of mask arrays.
        num_classes (int): Number of classes.

    Returns:
        np.array: Array of class frequencies.
    """
    class_counts = np.zeros(num_classes, dtype=int)
    for mask in masks:
        valid_mask = mask[~np.isnan(mask)]  # Ignore NaN values
        unique, counts = np.unique(valid_mask, return_counts=True)
        for cls, count in zip(unique, counts):
            cls = int(cls)  # Ensure class is an integer
            if 0 <= cls < num_classes:  # Check valid class range
                class_counts[cls] += count

    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels if total_pixels > 0 else np.zeros(num_classes)
    return np.round(class_frequencies, 8)  # Round to 8 decimal places


#%%
# Main workflow with logging and ProgressBar
def main_workflow(image_folders, mask_folders, num_classes=3, num_channels=5, log_file="dataset_log.txt"):
    """
    Main workflow to calculate class frequencies for the dataset.

    Args:
        image_folders (list of str): List of image folder paths.
        mask_folders (list of str): List of mask folder paths.
        num_classes (int): Number of classes for segmentation.
        num_channels (int): Number of channels in the images.
        log_file (str): Path to the log file.

    Returns:
        None
    """
    for image_folder, mask_folder in zip(image_folders, mask_folders):
        if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
            log_message(f"Skipping: {image_folder} or {mask_folder} (Missing folder)", log_file)
            continue

        log_message(f"Processing: {image_folder} and {mask_folder}", log_file)

        # Extract images and masks with ProgressBar
        images, masks = extract_images_and_masks(image_folder, mask_folder, num_channels=num_channels)

        # Log total dataset size
        total_size_before_subsampling = len(images)
        # log_message(f"Total dataset size before subsampling: {total_size_before_subsampling}", log_file)
        # log_message(f"Total dataset size after subsampling_site-specific: {total_size_before_subsampling}", log_file)
        log_message(f"Total dataset size after subsampling_single-site: {total_size_before_subsampling}", log_file)

        # Estimate original class frequencies
        original_class_frequencies = estimate_class_frequencies(masks, num_classes)
        # log_message(f"Original Class Frequencies: {original_class_frequencies.tolist()}", log_file)
        # log_message(f"Subsampled Class Frequencies: {original_class_frequencies.tolist()}", log_file)
        log_message(f"Subsampled Class Frequencies-sites: {original_class_frequencies.tolist()}", log_file)


#%%
# Run the Workflow
if __name__ == "__main__":
    start_time = time.time()
    # main_workflow(IMAGE_FOLDER, MASK_FOLDER, num_classes=3, num_channels=5, log_file="dataset_log.txt")  #low
    # main_workflow(IMAGE_FOLDER, MASK_FOLDER, num_classes=4, num_channels=5, log_file="dataset_log.txt")  #medium
    main_workflow(IMAGE_FOLDER, MASK_FOLDER, num_classes=5, num_channels=5, log_file="dataset_log.txt")  #dense
    print(f"Total time: {time.time() - start_time:.2f} seconds")

# %%
