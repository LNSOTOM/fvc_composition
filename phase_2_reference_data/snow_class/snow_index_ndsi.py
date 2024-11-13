#%%
import numpy as np
import os
import rasterio
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter

# The Normalized Difference Snow Index (NDSI) is used to identify snow cover
# def calculate_ndsi(green_band, swir_band):
#     """
#     Calculate the Normalized Difference Snow Index (NDSI).
#     NDSI = (Green - SWIR) / (Green + SWIR)
    
#     Parameters:
#     - green_band: The green band of the multispectral image.
#     - swir_band: The SWIR (shortwave infrared) band of the multispectral image.

#     Returns:
#     - ndsi: The calculated NDSI.
#     """
#     ndsi = (green_band - swir_band) / (green_band + swir_band)
#     return ndsi

def calculate_ndsi(green_band, swir_band):
    """
    Calculate the Normalized Difference Snow Index (NDSI).
    NDSI = (Green - SWIR) / (Green + SWIR)
    
    Handles cases where the denominator is zero or nan values exist.
    
    Parameters:
    - green_band: The green band of the multispectral image.
    - swir_band: The SWIR (shortwave infrared) band of the multispectral image.

    Returns:
    - ndsi: The calculated NDSI with proper handling of divide by zero or nan values.
    """
    # Avoid division by zero by setting invalid results to NaN
    denominator = (green_band + swir_band)
    ndsi = np.where(denominator != 0, (green_band - swir_band) / denominator, np.nan)
    
    return ndsi

def normalize_uint16_image(image):
    """
    Normalize a UInt16 image to the range [0, 1] for visualization.

    Parameters:
    - image: Input UInt16 image.

    Returns:
    - Normalized image in float32, with values between [0, 1].
    """
    image = image.astype(np.float32)
    max_value = np.max(image)  # Get the maximum value in the UInt16 image
    if max_value > 0:
        image /= max_value  # Normalize to [0, 1]
    return image

#%%
# Define the function to load and process a multispectral image with a percentile filter
def percentile_filter_noise_all_bands(image_path, percentile=99, footprint_size=3):
    """
    Load and process a multispectral image, applying a percentile filter with a defined footprint size to all bands.

    Parameters:
    - image_path: Path to the multispectral image file.
    - percentile: The percentile to use for filtering (default is 99).
    - footprint_size: The linear size of the square footprint in pixels (default is 3).

    Returns:
    - A numpy array of the processed multispectral image.
    """
    with rasterio.open(image_path) as src:
        # Read all bands
        image = src.read()  # Shape: (bands, height, width)

    # Transpose to (height, width, bands)
    image = image.transpose(1, 2, 0)

    processed_image = np.empty_like(image, dtype=np.float32)

    for i in range(image.shape[2]):  # Iterate over all bands
        # Apply the percentile filter with the specified footprint size
        processed_image[:, :, i] = percentile_filter(image[:, :, i], percentile, size=(footprint_size, footprint_size))

    return processed_image

# Define the linear contrast stretching function
def linear_contrast_stretching(image, min_percentile=1, max_percentile=99, stretch_min=0, stretch_max=1):
    """
    Apply linear contrast stretching to each band individually.

    Parameters:
    - image: Input image array with shape (height, width, bands).
    - min_percentile: Percentile value for minimum stretching (default: 1).
    - max_percentile: Percentile value for maximum stretching (default: 99).
    - stretch_min: Minimum value for stretching (default: 0).
    - stretch_max: Maximum value for stretching (default: 1).

    Returns:
    - Stretched image array.
    """
    stretched_image = np.zeros_like(image)
    for i in range(image.shape[2]):  # Loop through each band
        band_min = np.percentile(image[:, :, i], min_percentile)  # Percentile as minimum
        band_max = np.percentile(image[:, :, i], max_percentile)  # Percentile as maximum
        # Apply linear contrast stretching to the band
        stretched_image[:, :, i] = np.clip((image[:, :, i] - band_min) / (band_max - band_min) * (stretch_max - stretch_min) + stretch_min, stretch_min, stretch_max)
    return stretched_image

# Define the function to create a binary mask for NDSI (-1 to 1 range)
def create_binary_mask(data, threshold=0):
    """
    Create a binary mask based on NDSI values. 
    Values > threshold are considered as snow (1), others as non-snow (0).
    
    Parameters:
    - data: NDSI data ranging from -1 to 1.
    - threshold: Threshold value to classify snow (default: 0).

    Returns:
    - Binary mask where 1 indicates snow and 0 indicates non-snow.
    """
    snow_mask = np.zeros_like(data)
    snow_mask[data > threshold] = 1  # Set snow (NDSI > threshold)
    snow_mask[data <= threshold] = 0  # Set non-snow (NDSI <= threshold)
    
    return snow_mask


# Specify the path to your multispectral image
rgb_path = '/media/laura/Extreme SSD/code/ecosystem_composition/data_storage/snow/RGB_Landsat_Conger_Ice_Shelf_Prior.tif'
mutlispectral_path = '/media/laura/Extreme SSD/code/ecosystem_composition/data_storage/snow/Landsat8_Prior_Collapse_2019.tif'


# %%
## code 2 handling nan values
# Read the RGB images (UInt16)
rgb_image = tifffile.imread(rgb_path)

# Normalize the RGB image for display purposes
rgb_image_normalized = normalize_uint16_image(rgb_image)

# Process the multispectral image
processed_image = percentile_filter_noise_all_bands(mutlispectral_path)

# Normalize pixel values based on 1st percentile as minimum and 99th percentile as maximum
multispectral_img_stretched = linear_contrast_stretching(processed_image)

# Calculate the Normalized Difference Snow Index (NDSI)
# Green band index: 2 (band 3)
# SWIR band index: 5 (band 6)
snow_index = calculate_ndsi(processed_image[:, :, 2], processed_image[:, :, 5])

# Check if snow_index contains nan values and count them
nan_count = np.count_nonzero(np.isnan(snow_index))
print(f"Number of NaN values in NDSI: {nan_count}")

# Calculate the minimum and maximum values of the snow index, ignoring NaNs
min_value = np.nanmin(snow_index)
max_value = np.nanmax(snow_index)
mean_value = np.nanmean(snow_index)
std_value = np.nanstd(snow_index)

print("Minimum value (ignoring NaNs):", min_value)
print("Maximum value (ignoring NaNs):", max_value)
print("Mean value (ignoring NaNs):", mean_value)
print("Standard Deviation (ignoring NaNs):", std_value)

# Create the binary mask with NDSI threshold (default is 0)
binary_mask = create_binary_mask(snow_index, threshold=0)

# Plotting
fig, ax = plt.subplots(1, 4, figsize=(18, 12))  # Updated to 4 subplots

# Plot normalized RGB Imagery (normalized to [0, 1] for display)
im0 = ax[0].imshow(rgb_image_normalized)
ax[0].set_title(f'RGB Imagery \n')
cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

# Plot Multispectral with Percentile Filter and Linear Stretch
# im2 = ax[1].imshow(multispectral_img_stretched[:, :, [4, 2, 0]])
# ax[1].set_title('Multispectral \n')
# cbar2 = fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
im2 = ax[1].imshow(multispectral_img_stretched[:, :, [5, 4, 3]])
ax[1].set_title('Multispectral \n')
cbar2 = fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

# Plot NDSI
im3 = ax[2].imshow(snow_index, cmap='viridis')
ax[2].set_title('Snow Index - NDSI \n')
cbar3 = fig.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)

# Plot Binary Mask
im4 = ax[3].imshow(binary_mask, cmap='Greys_r')
ax[3].set_title('Snow Index Mask \n')
cbar4 = fig.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)

# Specify the output directory for saving the figure
output_dir = '/media/laura/Extreme SSD/code/ecosystem_composition/data_storage/snow'

plt.subplots_adjust(hspace=0.4, wspace=0.6)

# Save the figure with 300 DPI
output_filepath = os.path.join(output_dir, f'ndsi_swir_nir_red.png')
plt.savefig(output_filepath, dpi=300)

plt.show()
plt.close()  # Close the figure to free memory

# Output the shape of the processed image
print("Processed Image Shape:", processed_image.shape)

# Display statistics for the entire image
print("\nOverall Statistics for Processed Image:")
print("Min:", np.min(processed_image))
print("Max:", np.max(processed_image))
print("Mean:", np.mean(processed_image))
print("Std Dev:", np.std(processed_image))

# Display statistics for the snow index (NDSI)
print("\nSnow Index (NDSI) Statistics (Ignoring NaNs):")
print("Min:", min_value)
print("Max:", max_value)
print("Mean:", mean_value)
print("Std Dev:", std_value)


# %%
