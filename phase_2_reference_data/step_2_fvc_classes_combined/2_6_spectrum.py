
# %%
###test 1: one file
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t

# Load raster bands and classification mask
raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.22.tif'  # Replace with your raster file
mask_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.22.tif'  # Replace with your classification mask file


with rasterio.open(raster_path) as src:
    bands = src.read(out_dtype='float32')  # Shape: (5, height, width)
    bands[bands == src.nodata] = np.nan  # Handle nodata as NaN if applicable

with rasterio.open(mask_path) as src_mask:
    mask = src_mask.read(1).astype('float32')  # Read the mask
    mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

# Define class labels and colors
class_labels = {0: 'BE', 1: 'NPV', 2: 'PV', 3: 'SI'}
class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4']  # Colors for each class

# Wavelengths corresponding to the 5 bands
wavelengths = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands

# Initialize variables for mean and confidence intervals
mean_reflectance = {}
confidence_intervals = {}

# Confidence level for intervals
confidence_level = 0.95

# Process each class
for cls, cls_name in class_labels.items():
    class_mask = mask == cls  # Boolean mask for the current class
    
    # Skip if the class has no valid pixels
    if not np.any(class_mask):
        print(f"Class {cls_name} has no valid pixels. Skipping.")
        continue
    
    # Extract reflectance values for each band
    reflectance_values = [
        bands[band_idx][class_mask & ~np.isnan(bands[band_idx])]  # Exclude NaN
        for band_idx in range(bands.shape[0])
    ]
    
    # Calculate mean and confidence intervals for each band
    means = [np.nanmean(band) for band in reflectance_values]
    std_devs = [np.nanstd(band) for band in reflectance_values]
    n = [len(band) for band in reflectance_values]  # Number of valid pixels
    
    # Compute confidence intervals
    t_values = [t.ppf((1 + confidence_level) / 2, df=num - 1) if num > 1 else np.nan for num in n]
    cis = [t_val * (std / np.sqrt(num)) if num > 1 else np.nan for t_val, std, num in zip(t_values, std_devs, n)]
    
    # Store results
    mean_reflectance[cls_name] = means
    confidence_intervals[cls_name] = cis

# Plotting mean reflectance and confidence intervals
plt.figure(figsize=(12, 8))
for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]
        plt.plot(wavelengths, means, label=f'Class {cls_name}', color=color, linewidth=2)
        plt.fill_between(
            wavelengths,
            np.array(means) - np.array(cis),
            np.array(means) + np.array(cis),
            color=color,
            alpha=0.2
        )

plt.xticks(wavelengths, labels=[f"{wl} nm" for wl in wavelengths])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Mean Reflectance with Confidence Intervals Across Wavelengths')
plt.legend(title='Classes')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# %%
###test 2: plot in directories per file
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm  # Progress bar for better monitoring

# Define input directories
raster_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/'
mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/'

# Define class labels and colors
class_labels = {0: 'BE', 1: 'NPV', 2: 'PV', 3: 'SI'}
class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4']  # Colors for each class

# Wavelengths corresponding to the 5 bands
wavelengths = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands

# Confidence level for intervals
confidence_level = 0.95

# Helper function to extract numeric ID from filenames
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# Regular expressions for raster and mask files
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# List raster and mask files
raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith('.tif')}
mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith('.tif')}

# Match raster and mask files based on extracted IDs
common_ids = set(raster_files.keys()) & set(mask_files.keys())
if not common_ids:
    print("No matching raster-mask pairs found. Check file naming patterns.")
    exit()

# Prepare a progress bar
for file_id in tqdm(common_ids, desc="Processing raster-mask pairs"):
    raster_file = raster_files[file_id]
    mask_file = mask_files[file_id]

    raster_path = os.path.join(raster_dir, raster_file)
    mask_path = os.path.join(mask_dir, mask_file)

    print(f"Processing: Raster={raster_file}, Mask={mask_file}")

    with rasterio.open(raster_path) as src:
        bands = src.read(out_dtype='float32')  # Shape: (5, height, width)
        bands[bands == src.nodata] = np.nan  # Handle nodata as NaN if applicable

    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1).astype('float32')  # Read the mask
        mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

    # Apply a global NaN mask to all bands
    valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands), axis=0)

    # Initialize storage for statistics
    mean_reflectance = {}
    confidence_intervals = {}

    # Process each class
    for cls, cls_name in class_labels.items():
        class_mask = (mask == cls) & valid_pixels_mask  # Combine class and valid pixel masks

        # Skip if the class has no valid pixels
        if not np.any(class_mask):
            continue

        # Use NumPy advanced indexing to extract all bands for the class
        reflectance_values = bands[:, class_mask]

        # Calculate mean and confidence intervals for each band
        means = np.nanmean(reflectance_values, axis=1)
        std_devs = np.nanstd(reflectance_values, axis=1)
        n = reflectance_values.shape[1]  # Number of valid pixels

        # Compute confidence intervals
        t_value = t.ppf((1 + confidence_level) / 2, df=n - 1) if n > 1 else np.nan
        cis = t_value * (std_devs / np.sqrt(n)) if n > 1 else np.full_like(means, np.nan)

        # Store results
        mean_reflectance[cls_name] = means
        confidence_intervals[cls_name] = cis

    # Plotting mean reflectance and confidence intervals
    plt.figure(figsize=(12, 8))
    for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
        if cls_name in mean_reflectance:
            means = mean_reflectance[cls_name]
            cis = confidence_intervals[cls_name]
            plt.plot(wavelengths, means, label=f'Class {cls_name}', color=color, linewidth=2)
            plt.fill_between(
                wavelengths,
                means - cis,
                means + cis,
                color=color,
                alpha=0.2
            )

    plt.xticks(wavelengths, labels=[f"{wl} nm" for wl in wavelengths])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'Mean Reflectance with Confidence Intervals for Raster ID: {file_id}')
    plt.legend(title='Classes')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Save the plot to file
    output_plot_path = os.path.join(raster_dir, f"reflectance_plot_{file_id}.png")
    plt.savefig(output_plot_path, dpi=150)
    plt.close()

    print(f"Saved plot for Raster ID: {file_id} to {output_plot_path}")


# %%
###test 3 one plot per directory and site-specific
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm  # Progress bar for better monitoring

# Define input directories
## low site
raster_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked'
mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster'
# medium site
# raster_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/'
# mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/'
# # dense
# raster_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/stacked'
# mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster'

# Define the output directory
output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define class labels and colors
class_labels = {0: 'BE', 1: 'NPV', 2: 'PV', 3: 'SI', 4: 'WI'}
class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # Colors for each class

# Wavelengths corresponding to the 5 bands
wavelengths = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands

# Confidence level for intervals
confidence_level = 0.95

# Helper function to extract numeric ID from filenames
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# Regular expressions for raster and mask files
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# List raster and mask files
raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith('.tif')}
mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith('.tif')}

# Match raster and mask files based on extracted IDs
common_ids = set(raster_files.keys()) & set(mask_files.keys())
if not common_ids:
    print("No matching raster-mask pairs found. Check file naming patterns.")
    exit()

# Initialize storage for aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Prepare a progress bar
for file_id in tqdm(common_ids, desc="Processing raster-mask pairs"):
    raster_file = raster_files[file_id]
    mask_file = mask_files[file_id]

    raster_path = os.path.join(raster_dir, raster_file)
    mask_path = os.path.join(mask_dir, mask_file)

    print(f"Processing: Raster={raster_file}, Mask={mask_file}")

    with rasterio.open(raster_path) as src:
        bands = src.read(out_dtype='float32')  # Shape: (5, height, width)
        bands[bands == src.nodata] = np.nan  # Handle nodata as NaN if applicable

    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1).astype('float32')  # Read the mask
        mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

    # Apply a global NaN mask to all bands
    valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands), axis=0)

    # Process each class
    for cls, cls_name in class_labels.items():
        class_mask = (mask == cls) & valid_pixels_mask  # Combine class and valid pixel masks

        # Skip if the class has no valid pixels
        if not np.any(class_mask):
            continue

        # Use NumPy advanced indexing to extract all bands for the class
        reflectance_values = bands[:, class_mask]
        aggregated_reflectance[cls_name].append(reflectance_values)

# Compute overall mean reflectance and confidence intervals
mean_reflectance = {}
confidence_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue  # Skip classes with no data

    # Stack all reflectance values for the class across files
    reflectance_data = np.hstack(reflectance_list)  # Shape: (bands, total_pixels)

    # Calculate mean and confidence intervals for each band
    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)
    n = reflectance_data.shape[1]  # Total number of pixels

    # Compute confidence intervals
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1) if n > 1 else np.nan
    cis = t_value * (std_devs / np.sqrt(n)) if n > 1 else np.full_like(means, np.nan)

    mean_reflectance[cls_name] = means
    confidence_intervals[cls_name] = cis

# Plotting mean reflectance and confidence intervals
plt.figure(figsize=(12, 8))
for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]
        plt.plot(wavelengths, means, label=f'{cls_name}', color=color, linewidth=2)
        plt.fill_between(
            wavelengths,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2
        )

plt.xticks(wavelengths, labels=[f"{wl}" for wl in wavelengths])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Mean Reflectance with Confidence Intervals Across All Raster Files')
plt.legend(title='Classes')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot to the specified directory
output_plot_path = os.path.join(output_dir, "mean_reflectance_low.png")
# output_plot_path = os.path.join(output_dir, "mean_reflectance_medium.png")
# output_plot_path = os.path.join(output_dir, "mean_reflectance_dense.png")
plt.savefig(output_plot_path, dpi=150)
plt.show()

print(f"Saved overall plot to {output_plot_path}")

# %%
###test 4 : multiple folders for sites
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm  # Progress bar for better monitoring

# List of input directories
input_dirs = [
    {
        "raster_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked",
        "mask_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster",
    },
    {
        "raster_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/",
        "mask_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/",
    },
    {
        "raster_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/stacked",
        "mask_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster",
    },
]

# Define the output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]  # Colors for each class

# Wavelengths corresponding to the 5 bands
wavelengths = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands

# Confidence level for intervals
confidence_level = 0.95

# Helper function to extract numeric ID from filenames
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# Regular expressions for raster and mask files
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# Initialize storage for aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Loop through all directories
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    # List raster and mask files
    raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    # Match raster and mask files based on extracted IDs
    common_ids = set(raster_files.keys()) & set(mask_files.keys())
    if not common_ids:
        print(f"No matching raster-mask pairs found in {raster_dir} and {mask_dir}.")
        continue

    # Process each raster-mask pair
    for file_id in tqdm(common_ids, desc=f"Processing files in {raster_dir}"):
        raster_file = raster_files[file_id]
        mask_file = mask_files[file_id]

        raster_path = os.path.join(raster_dir, raster_file)
        mask_path = os.path.join(mask_dir, mask_file)

        with rasterio.open(raster_path) as src:
            bands = src.read(out_dtype="float32")  # Shape: (5, height, width)
            bands[bands == src.nodata] = np.nan  # Handle nodata as NaN if applicable

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")  # Read the mask
            mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

        # Apply a global NaN mask to all bands
        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands), axis=0)

        # Process each class
        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask  # Combine class and valid pixel masks

            # Skip if the class has no valid pixels
            if not np.any(class_mask):
                continue

            # Use NumPy advanced indexing to extract all bands for the class
            reflectance_values = bands[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute overall mean reflectance and confidence intervals
mean_reflectance = {}
confidence_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue  # Skip classes with no data

    # Stack all reflectance values for the class across files
    reflectance_data = np.hstack(reflectance_list)  # Shape: (bands, total_pixels)

    # Calculate mean and confidence intervals for each band
    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)
    n = reflectance_data.shape[1]  # Total number of pixels

    # Compute confidence intervals
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1) if n > 1 else np.nan
    cis = t_value * (std_devs / np.sqrt(n)) if n > 1 else np.full_like(means, np.nan)

    mean_reflectance[cls_name] = means
    confidence_intervals[cls_name] = cis

# Plotting mean reflectance and confidence intervals
plt.figure(figsize=(12, 8))
for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]
        plt.plot(wavelengths, means, label=f"Class {cls_name}", color=color, linewidth=2)
        plt.fill_between(
            wavelengths,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2
        )

plt.xticks(wavelengths, labels=[f"{wl} nm" for wl in wavelengths])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Mean Reflectance with Confidence Intervals Across All Folders")
plt.legend(title="Classes")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save the plot to the specified directory
output_plot_path = os.path.join(output_dir, "mean_reflectance_sites.png")
plt.savefig(output_plot_path, dpi=150)
plt.show()

print(f"Saved overall plot to {output_plot_path}")

# %%
##test 5: with stats
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm  # Progress bar for better monitoring

# List of input directories
input_dirs = [
    {
        "raster_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked",
        "mask_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster",
    },
    {
        "raster_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/",
        "mask_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/",
    },
    {
        "raster_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/stacked",
        "mask_dir": "/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster",
    },
]

# Define the output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]  # Colors for each class

# Wavelengths corresponding to the 5 bands
wavelengths = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands

# Confidence level for intervals
confidence_level = 0.95

# Helper function to extract numeric ID from filenames
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# Regular expressions for raster and mask files
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# Initialize storage for aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Loop through all directories
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    # List raster and mask files
    raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    # Match raster and mask files based on extracted IDs
    common_ids = set(raster_files.keys()) & set(mask_files.keys())
    if not common_ids:
        print(f"No matching raster-mask pairs found in {raster_dir} and {mask_dir}.")
        continue

    # Process each raster-mask pair
    for file_id in tqdm(common_ids, desc=f"Processing files in {raster_dir}"):
        raster_file = raster_files[file_id]
        mask_file = mask_files[file_id]

        raster_path = os.path.join(raster_dir, raster_file)
        mask_path = os.path.join(mask_dir, mask_file)

        with rasterio.open(raster_path) as src:
            bands = src.read(out_dtype="float32")  # Shape: (5, height, width)
            bands[bands == src.nodata] = np.nan  # Handle nodata as NaN if applicable

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")  # Read the mask
            mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

        # Apply a global NaN mask to all bands
        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands), axis=0)

        # Process each class
        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask  # Combine class and valid pixel masks

            # Skip if the class has no valid pixels
            if not np.any(class_mask):
                continue

            # Use NumPy advanced indexing to extract all bands for the class
            reflectance_values = bands[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute overall mean reflectance and confidence intervals
mean_reflectance = {}
confidence_intervals = {}
standard_deviations = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue  # Skip classes with no data

    # Stack all reflectance values for the class across files
    reflectance_data = np.hstack(reflectance_list)  # Shape: (bands, total_pixels)

    # Calculate mean and confidence intervals for each band
    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)
    n = reflectance_data.shape[1]  # Total number of pixels

    # Compute confidence intervals
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1) if n > 1 else np.nan
    cis = t_value * (std_devs / np.sqrt(n)) if n > 1 else np.full_like(means, np.nan)

    mean_reflectance[cls_name] = means
    confidence_intervals[cls_name] = cis
    standard_deviations[cls_name] = std_devs

    # Print the statistics for the class
    print(f"\nClass: {cls_name}")
    print("Wavelength (nm):", wavelengths)
    print("Mean Reflectance:", means)
    print("Standard Deviation:", std_devs)
    print("Confidence Intervals:", cis)

# Plotting mean reflectance and confidence intervals
plt.figure(figsize=(12, 8))
for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]
        plt.plot(wavelengths, means, label=f"{cls_name}", color=color, linewidth=2)
        plt.fill_between(
            wavelengths,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2
        )

plt.xticks(wavelengths, labels=[f"{wl}" for wl in wavelengths])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Mean Reflectance with Confidence Intervals Across All Folders")
plt.legend(title="Classes")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save the plot to the specified directory
output_plot_path = os.path.join(output_dir, "mean_reflectance_sites.png")
plt.savefig(output_plot_path, dpi=150)
plt.show()

print(f"\nSaved overall plot to {output_plot_path}")
