
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
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
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
plt.title("Mean reflectance with confidence intervals across all sites")
plt.legend(title="Classes:")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save the plot to the specified directory
output_plot_path = os.path.join(output_dir, "mean_reflectance_sites.png")
plt.savefig(output_plot_path, dpi=150)
plt.show()

print(f"\nSaved overall plot to {output_plot_path}")

# %%
###test 6: wavelenght values (option A)
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
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]  # Colors for each class

# Wavelengths corresponding to the 5 bands
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range

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
            bands_data = src.read(out_dtype="float32")  # Shape: (5, height, width)
            bands_data[bands_data == src.nodata] = np.nan  # Handle nodata as NaN if applicable

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")  # Read the mask
            mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

        # Apply a global NaN mask to all bands
        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        # Process each class
        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask  # Combine class and valid pixel masks

            # Skip if the class has no valid pixels
            if not np.any(class_mask):
                continue

            # Use NumPy advanced indexing to extract all bands for the class
            reflectance_values = bands_data[:, class_mask]
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
        plt.plot(bands, means, label=f"{cls_name}", color=color, linewidth=2)
        plt.fill_between(
            bands,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2
        )

plt.xticks(x_axis_wavelengths, labels=[f"{wl}" for wl in x_axis_wavelengths])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Mean reflectance with confidence intervals across all sites")
plt.legend(title="Classes:")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save the plot to the specified directory
output_plot_path = os.path.join(output_dir, "mean_reflectance_sites_nm_a.png")
plt.savefig(output_plot_path, dpi=150)
plt.show()

print(f"\nSaved overall plot to {output_plot_path}")

# %%
###test 6: wavelenght values (option B)
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
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]  # Colors for each class

# Wavelengths corresponding to the 5 bands
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range

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
            bands_data = src.read(out_dtype="float32")  # Shape: (5, height, width)
            bands_data[bands_data == src.nodata] = np.nan  # Handle nodata as NaN if applicable

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")  # Read the mask
            mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

        # Apply a global NaN mask to all bands
        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        # Process each class
        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask  # Combine class and valid pixel masks

            # Skip if the class has no valid pixels
            if not np.any(class_mask):
                continue

            # Use NumPy advanced indexing to extract all bands for the class
            reflectance_values = bands_data[:, class_mask]
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

# Combine actual and extended x-axis values for plotting
xticks_combined = sorted(set(bands + x_axis_wavelengths))

# Plotting mean reflectance and confidence intervals
plt.figure(figsize=(12, 8))
for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]
        plt.plot(bands, means, label=f"{cls_name}", color=color, linewidth=2)
        plt.fill_between(
            bands,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2
        )

# Add both sets of x-axis values
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Mean reflectance with confidence intervals across all sites")
plt.legend(title="Classes:", loc="upper left")  # Legend positioned in top-left corner
# plt.legend(title="Classes:")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save the plot to the specified directory
output_plot_path = os.path.join(output_dir, "mean_reflectance_sites_nm_b.png")
plt.savefig(output_plot_path, dpi=150)
plt.show()

print(f"\nSaved overall plot to {output_plot_path}")


# %%
### test 7: imporve plot 
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm  # Progress bar for better monitoring
from matplotlib.ticker import FuncFormatter


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
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]  # Colors for each class

# Wavelengths corresponding to the 5 bands
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range

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
            bands_data = src.read(out_dtype="float32")  # Shape: (5, height, width)
            bands_data[bands_data == src.nodata] = np.nan  # Handle nodata as NaN if applicable

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")  # Read the mask
            mask[mask == src_mask.nodata] = np.nan  # Handle nodata as NaN if applicable

        # Apply a global NaN mask to all bands
        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        # Process each class
        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask  # Combine class and valid pixel masks

            # Skip if the class has no valid pixels
            if not np.any(class_mask):
                continue

            # Use NumPy advanced indexing to extract all bands for the class
            reflectance_values = bands_data[:, class_mask]
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

# Combine extended and actual wavelengths for x-axis
xticks_combined = sorted(set(x_axis_wavelengths + bands))

# Plotting mean reflectance and confidence intervals
plt.figure(figsize=(12, 8))

# Custom formatter for rounding ticks to 2 decimal places
formatter = FuncFormatter(lambda x, _: f"{x:.2f}")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

# Set tick parameters to increase size of x-axis and y-axis ticks
plt.tick_params(axis='both', which='major', labelsize=12)  # Adjust labelsize as needed
plt.tick_params(axis='both', which='minor', labelsize=12)

# Calculate the upper limit for the y-axis
y_max = max(
    max(np.array(mean_reflectance[cls_name]) + np.array(confidence_intervals[cls_name]))
    for cls_name in mean_reflectance
)

# Set the y-axis to start at 0 and end just above the maximum data value
# plt.ylim(0, y_max + 0.01)  # Add a small buffer to the upper limit for better visualization
y_ticks = np.arange(0, y_max + 0.03, 0.03)  # Define ticks from 0 to y_max with a step of 0.02
plt.ylim(0, y_ticks[-1])  # Set y-axis limits to the defined range
plt.yticks(y_ticks)  # Apply the y-axis ticks


for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]
        plt.plot(bands, means, label=f"{cls_name}", color=color, linewidth=2)
        plt.fill_between(
            bands,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2
        )

# Add ticks for both actual and extended wavelengths
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])
# Draw vertical gridlines for the actual wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# plt.xlabel("Wavelength (nm)", fontsize=14)
# plt.ylabel("Reflectance", fontsize=14)
# plt.title("Mean reflectance with confidence intervals across all sites")
plt.legend(title="Classes:", loc="upper left")  # Legend positioned in top-left corner
plt.grid(axis="x", linestyle="--", alpha=0.0)  # Only gridlines for actual wavelengths

# Save the plot to the specified directory
output_plot_path = os.path.join(output_dir, "mean_reflectance_sites_bands_t.png")
plt.savefig(output_plot_path, dpi=150)
plt.show()

print(f"\nSaved overall plot to {output_plot_path}")


# %%
###test 8
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range
confidence_level = 0.95

# Helper function to extract IDs
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# File patterns
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# Aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    common_ids = set(raster_files.keys()) & set(mask_files.keys())
    if not common_ids:
        print(f"No matching raster-mask pairs found in {raster_dir} and {mask_dir}.")
        continue

    for file_id in tqdm(common_ids, desc=f"Processing files in {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[file_id])
        mask_path = os.path.join(mask_dir, mask_files[file_id])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        # Detect available classes dynamically
        detected_classes = np.unique(mask[~np.isnan(mask)]).astype(int)
        detected_classes = [cls for cls in detected_classes if cls in class_labels.keys()]

        for cls in detected_classes:
            cls_name = class_labels[cls]
            class_mask = (mask == cls) & valid_pixels_mask

            if not np.any(class_mask):
                print(f"No valid pixels for class {cls_name} in file {file_id}")
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and confidence intervals
mean_reflectance = {}
confidence_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        print(f"No data for class {cls_name}.")
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)
    n = reflectance_data.shape[1]

    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1) if n > 1 else np.nan
    cis = t_value * (std_devs / np.sqrt(n)) if n > 1 else np.full_like(means, np.nan)

    mean_reflectance[cls_name] = means
    confidence_intervals[cls_name] = cis

# Plotting
plt.figure(figsize=(12, 8))

for idx, (cls_name, color) in enumerate(zip(class_labels.values(), class_colors)):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]
        plt.plot(bands, means, label=f"{cls_name}", color=color, linewidth=2)
        plt.fill_between(
            bands,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2,
            label=f"{cls_name} CI"
        )

# Add extended x-axis ticks
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for the actual wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

plt.legend(title="Classes", loc="upper left")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Mean Reflectance with Confidence Intervals")
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_ci_extended_xaxis.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")


# %%
###test 9
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths and extended x-axis
bands = [475, 560, 668, 717, 842]
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]
confidence_level = 0.95

# Helper function to extract IDs
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# File patterns
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# Aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    common_ids = set(raster_files.keys()) & set(mask_files.keys())
    if not common_ids:
        print(f"No matching raster-mask pairs found in {raster_dir} and {mask_dir}.")
        continue

    for file_id in tqdm(common_ids, desc=f"Processing files in {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[file_id])
        mask_path = os.path.join(mask_dir, mask_files[file_id])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and confidence intervals
mean_reflectance = {}
confidence_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        print(f"No data for class {cls_name}.")
        continue

    reflectance_data = np.hstack(reflectance_list)
    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)
    n = reflectance_data.shape[1]

    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1) if n > 1 else np.nan
    cis = t_value * (std_devs / np.sqrt(n)) if n > 1 else np.full_like(means, np.nan)

    mean_reflectance[cls_name] = means
    confidence_intervals[cls_name] = cis

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]

        # Plot mean reflectance
        plt.plot(
            bands,
            means,
            label=f"{cls_name}",
            color=color,
            linewidth=2,
            marker='o',
            markersize=8,
        )

        # Plot vertical error bars for confidence intervals
        plt.errorbar(
            bands,
            means,
            yerr=cis,
            fmt='o',
            color=color,
            ecolor='black',
            elinewidth=1.5,
            capsize=5,
        )

# Add extended x-axis ticks
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with Confidence Intervals", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12, title_fontsize=14, frameon=True, edgecolor="gray")

# Gridlines and save plot
plt.grid(axis="both", linestyle="--", alpha=0.5)
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_error_bars_corrected.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
###test 10
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths and extended x-axis
bands = [475, 560, 668, 717, 842]
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]
confidence_level = 0.95

# Helper function to extract IDs
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# File patterns
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# Aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    common_ids = set(raster_files.keys()) & set(mask_files.keys())
    if not common_ids:
        print(f"No matching raster-mask pairs found in {raster_dir} and {mask_dir}.")
        continue

    for file_id in tqdm(common_ids, desc=f"Processing files in {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[file_id])
        mask_path = os.path.join(mask_dir, mask_files[file_id])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and confidence intervals
mean_reflectance = {}
confidence_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        print(f"No data for class {cls_name}.")
        continue

    reflectance_data = np.hstack(reflectance_list)
    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)
    n = reflectance_data.shape[1]

    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1) if n > 1 else np.nan
    cis = t_value * (std_devs / np.sqrt(n)) if n > 1 else np.full_like(means, np.nan)

    mean_reflectance[cls_name] = means
    confidence_intervals[cls_name] = cis

# Plotting with shaded confidence intervals
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]

        # Plot mean reflectance with a dashed line
        plt.plot(
            bands,
            means,
            label=f"{cls_name}",
            color=color,
            linewidth=2,
            # linestyle='--',  # Dashed line for mean reflectance
            # marker='o',
            # markersize=8,
        )

        # Plot shaded confidence interval with increased transparency
        plt.fill_between(
            bands,
            means - cis,
            means + cis,
            color=color,
            alpha=0.15,  # Increased transparency for shading
            label=f"{cls_name} CI",
        )

        # Add error bars for clarity
        plt.errorbar(
            bands,
            means,
            yerr=cis,
            fmt='o',
            color=color,
            ecolor='black',  # Black error bars for contrast
            elinewidth=1,
            capsize=3,
        )

# Add extended x-axis ticks
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with Shaded Confidence Intervals", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12, title_fontsize=14, frameon=True, edgecolor="gray")

# Gridlines and save plot
plt.grid(axis="both", linestyle="--", alpha=0.5)
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_distinct_cis.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")


# %%##test 11
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths and extended x-axis
bands = [475, 560, 668, 717, 842]
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]
confidence_level = 0.95

# Helper function to extract IDs
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# File patterns
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# Aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    common_ids = set(raster_files.keys()) & set(mask_files.keys())
    if not common_ids:
        print(f"No matching raster-mask pairs found in {raster_dir} and {mask_dir}.")
        continue

    for file_id in tqdm(common_ids, desc=f"Processing files in {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[file_id])
        mask_path = os.path.join(mask_dir, mask_files[file_id])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and confidence intervals for each wavelength
mean_reflectance = {}
confidence_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        print(f"No data for class {cls_name}.")
        continue

    # Combine reflectance data for this class from all rasters
    reflectance_data = np.hstack(reflectance_list)  # Shape: [bands, total_pixels]

    # Initialize storage for means and CIs for this class
    class_means = []
    class_cis = []

    for i, wavelength in enumerate(bands):  # Iterate over each wavelength
        band_data = reflectance_data[i, :]  # Data for the current band
        band_data = band_data[~np.isnan(band_data)]  # Remove NaNs

        # Compute mean and CI if there are enough valid pixels
        if len(band_data) > 1:
            mean = np.nanmean(band_data)
            std_dev = np.nanstd(band_data)
            n = len(band_data)

            t_value = t.ppf((1 + confidence_level) / 2, df=n - 1)
            ci = t_value * (std_dev / np.sqrt(n))
        else:
            mean, ci = np.nan, np.nan  # Not enough data to compute

        class_means.append(mean)
        class_cis.append(ci)

    # Store the results for this class
    mean_reflectance[cls_name] = np.array(class_means)
    confidence_intervals[cls_name] = np.array(class_cis)

# Plotting with shaded confidence intervals
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]

        # Plot mean reflectance with a dashed line
        plt.plot(
            bands,
            means,
            label=f"{cls_name}",
            color=color,
            linewidth=2,
        )

        # Plot shaded confidence interval with increased transparency
        plt.fill_between(
            bands,
            means - cis,
            means + cis,
            color=color,
            alpha=0.2,  # Transparency for shading
            label=f"{cls_name} CI",
        )

        # Add error bars for clarity
        plt.errorbar(
            bands,
            means,
            yerr=cis,
            fmt='o',
            color=color,
            ecolor='black',  # Black error bars for contrast
            elinewidth=1,
            capsize=3,
        )

# Add extended x-axis ticks
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with Shaded Confidence Intervals", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12, title_fontsize=14, frameon=True, edgecolor="gray")

# Gridlines and save plot
plt.grid(axis="both", linestyle="--", alpha=0.5)
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_cis.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")


# %%
##test 12
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths and extended x-axis
bands = [475, 560, 668, 717, 842]
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]
confidence_level = 0.95

# Helper function to extract IDs
def extract_id(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

# File patterns
raster_pattern = r"tiles_multispectral\.(\d+)\.tif"
mask_pattern = r"mask_fvc_3072\.(\d+)\.tif"

# Aggregated reflectance data
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {extract_id(f, raster_pattern): f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {extract_id(f, mask_pattern): f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    common_ids = set(raster_files.keys()) & set(mask_files.keys())
    if not common_ids:
        print(f"No matching raster-mask pairs found in {raster_dir} and {mask_dir}.")
        continue

    for file_id in tqdm(common_ids, desc=f"Processing files in {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[file_id])
        mask_path = os.path.join(mask_dir, mask_files[file_id])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and confidence intervals for each wavelength
mean_reflectance = {}
confidence_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        print(f"No data for class {cls_name}.")
        continue

    # Combine reflectance data for this class from all rasters
    reflectance_data = np.hstack(reflectance_list)  # Shape: [bands, total_pixels]

    # Initialize storage for means and CIs for this class
    class_means = []
    class_cis = []

    for i, wavelength in enumerate(bands):  # Iterate over each wavelength
        band_data = reflectance_data[i, :]  # Data for the current band
        band_data = band_data[~np.isnan(band_data)]  # Remove NaNs

        # Compute mean and CI if there are enough valid pixels
        if len(band_data) > 1:
            mean = np.nanmean(band_data)
            std_dev = np.nanstd(band_data)
            n = len(band_data)

            t_value = t.ppf((1 + confidence_level) / 2, df=n - 1)
            ci = t_value * (std_dev / np.sqrt(n))

            # Debugging: Print values for each wavelength and class
            print(f"Class: {cls_name}, Wavelength: {wavelength}, Mean: {mean:.10f}, CI: {ci:.10f}, n: {n}")
        else:
            mean, ci = np.nan, np.nan  # Not enough data to compute

        class_means.append(mean)
        class_cis.append(ci)

    # Store the results for this class
    mean_reflectance[cls_name] = np.array(class_means)
    confidence_intervals[cls_name] = np.array(class_cis)

# Plotting Mean Reflectance
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]

        # Plot mean reflectance
        plt.plot(
            bands,
            means,
            label=f"{cls_name}",
            color=color,
            linewidth=2,
        )

# Add extended x-axis ticks
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance for Each Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12, title_fontsize=14, frameon=True, edgecolor="gray")

# Save mean plot
mean_plot_path = os.path.join(output_dir, "mean_reflectance_only.png")
plt.savefig(mean_plot_path, dpi=300)
plt.show()

print(f"Saved mean reflectance plot to {mean_plot_path}")

# Plotting Confidence Intervals
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        cis = confidence_intervals[cls_name]

        # Plot shaded confidence intervals
        plt.fill_between(
            bands,
            means - cis,
            means + cis,
            color=color,
            alpha=0.3,
            label=f"{cls_name} 95% CI",
        )

# Add extended x-axis ticks
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("95% Confidence Intervals for Reflectance", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12, title_fontsize=14, frameon=True, edgecolor="gray")

# Save CI plot
ci_plot_path = os.path.join(output_dir, "confidence_intervals_only.png")
plt.savefig(ci_plot_path, dpi=300)
plt.show()

print(f"Saved confidence intervals plot to {ci_plot_path}")


# %%
##test 13
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar for better monitoring

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths corresponding to the 5 bands
bands = [475, 560, 668, 717, 842]
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]

# Initialize storage for aggregated reflectance data per class
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_file)
        mask_path = os.path.join(mask_dir, mask_file)

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals per class
mean_reflectance = {}
ci_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)  # Stack reflectance data for the class

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)
    n = reflectance_data.shape[1]

    # Compute prediction intervals (95% confidence level)
    interval = 1.96 * std_devs
    lower, upper = means - interval, means + interval

    mean_reflectance[cls_name] = means
    ci_intervals[cls_name] = (lower, upper)

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        lower_ci, upper_ci = ci_intervals[cls_name]

        # Plot mean reflectance
        plt.plot(
            bands,
            means,
            label=f"{cls_name} Mean",
            color=color,
            linewidth=2,
        )

        # Plot vertical dashed lines for prediction intervals for this class
        for i, wavelength in enumerate(bands):
            plt.plot(
                [wavelength, wavelength],  # Fixed x value (wavelength)
                [lower_ci[i], upper_ci[i]],  # CI range for y-axis
                linestyle="--",
                color=color,
                alpha=0.8,
                linewidth=1,
            )

# Add extended x-axis ticks
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Labels and title
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save the plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_prediction_intervals_per_class.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
##test 14
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_file)
        mask_path = os.path.join(mask_dir, mask_file)

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
mean_reflectance = {}
prediction_intervals = {}

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    mean_reflectance[cls_name] = means
    prediction_intervals[cls_name] = (lower, upper)

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name in mean_reflectance:
        means = mean_reflectance[cls_name]
        lower, upper = prediction_intervals[cls_name]

        # Plot mean reflectance
        plt.errorbar(
            bands,  # X-axis (wavelengths)
            means,  # Y-axis (mean reflectance)
            yerr=(upper - means),  # Error bars (upper bound)
            fmt="o-",  # Line with circular markers
            color=color,
            capsize=3,  # Size of caps on error bars
            label=f"{cls_name} Mean with CI",
        )

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_errorbars.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
####TEST 15
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table in your console
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance with error bars
    plt.errorbar(
        wavelengths,  # X-axis (wavelengths)
        means,  # Y-axis (mean reflectance)
        yerr=(upper_ci - means),  # Error bars (upper bound)
        fmt="o-",  # Line with circular markers
        color=color,
        capsize=3,  # Size of caps on error bars
        label=f"{cls_name} Mean with CI",
    )

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_errorbars.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
##TEST 16
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as dashed lines
    plt.plot(
        wavelengths,
        lower_ci,
        "--",
        color=color,
        linewidth=1,
        label=f"{cls_name} Lower CI",
    )
    plt.plot(
        wavelengths,
        upper_ci,
        "--",
        color=color,
        linewidth=1,
        label=f"{cls_name} Upper CI",
    )

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_full_and_dashed_lines.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
###TEST 17
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
        label=f"{cls_name} 95% CI",
    )

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_ci.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
##TEST 18
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

# Check if class exists in results_df
for cls_name, color in zip(class_labels.values(), class_colors):
    if cls_name not in results_df["Class"].unique():
        print(f"Class {cls_name} not found in results_df. Skipping...")
        continue

    # Filter data for the class
    class_data = results_df[results_df["Class"] == cls_name]
    if class_data.empty:
        print(f"No data available for class {cls_name}.")
        continue

    # Extract data for plotting
    wavelengths = class_data["Wavelength"].values
    means = class_data["Mean Reflectance"].values
    lower_ci = class_data["Lower CI"].values
    upper_ci = class_data["Upper CI"].values

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
        label=f"{cls_name} 95% CI",
    )

    # Add vertical dashed lines for confidence intervals
    for i, wl in enumerate(wavelengths):
        plt.vlines(
            x=wl,
            ymin=lower_ci[i],
            ymax=upper_ci[i],
            color=color,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_ci_and_vlines.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")


# %%
##TEST 19
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
        label=f"{cls_name} 95% CI",
    )

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_ci.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
###TEST 20
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
    )

    # Dashed lines for lower and upper CIs
    plt.plot(
        wavelengths,
        lower_ci,
        "--",
        color=color,
        linewidth=1,
        label=f"{cls_name} Lower CI",
    )
    plt.plot(
        wavelengths,
        upper_ci,
        "--",
        color=color,
        linewidth=1,
        label=f"{cls_name} Upper CI",
    )

# Add ticks for actual and extended wavelengths
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Labels, title, and legend
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.legend(title="Classes", loc="upper left", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_ci_sites.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
##twst 21
# Updated Plotting Code
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
    )

    # Dashed lines for lower and upper CIs (grouped in legend)
    plt.plot(
        wavelengths,
        lower_ci,
        "--",
        color=color,
        linewidth=1,
    )
    plt.plot(
        wavelengths,
        upper_ci,
        "--",
        color=color,
        linewidth=1,
    )

# Add ticks for actual and extended wavelengths
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Customize y-axis to display two decimal places
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
# Set tick parameters to increase size of x-axis and y-axis ticks
plt.tick_params(axis='both', which='major', labelsize=12)  # Adjust labelsize as needed
plt.tick_params(axis='both', which='minor', labelsize=12)

# Labels, title, and legend
# plt.xlabel("Wavelength (nm)", fontsize=14)
# plt.ylabel("Reflectance", fontsize=14)
# plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)

# Custom legend: Add single entry for CIs per class
handles, labels = plt.gca().get_legend_handles_labels()
ci_handles = [
    plt.Line2D([0], [0], linestyle="--", color=color, linewidth=1) for color in class_colors
]
ci_labels = [f"{cls_name} CI" for cls_name in class_labels.values()]
combined_handles = handles + ci_handles
combined_labels = labels + ci_labels
plt.legend(combined_handles, combined_labels, title="Classes:", loc="upper left", fontsize=12)

# Gridlines
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_ci_sites.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
###test 22
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
    )

# Combine extended and actual wavelengths for x-axis
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for the actual wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Customize y-axis to display two decimal places
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
# Set tick parameters to increase size of x-axis and y-axis ticks
plt.tick_params(axis='both', which='major', labelsize=12)  # Adjust labelsize as needed
plt.tick_params(axis='both', which='minor', labelsize=12)

# Custom legend: Add single entry for CIs per class
handles, labels = plt.gca().get_legend_handles_labels()
ci_handles = [
    plt.Line2D([0], [0], linestyle="--", color=color, linewidth=1) for color in class_colors
]
ci_labels = [f"{cls_name} CI" for cls_name in class_labels.values()]
combined_handles = handles + ci_handles
combined_labels = labels + ci_labels
plt.legend(combined_handles, combined_labels, title="Classes:", loc="upper left", fontsize=12)

# Labels, title, and grid
# plt.xlabel("Wavelength (nm)", fontsize=14)
# plt.ylabel("Reflectance", fontsize=14)
# plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
# plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_ci_custom_legend.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
###test 23
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
    )

    # Add markers for lower and upper bounds
    plt.scatter(wavelengths, lower_ci, color=color, marker="v", label=f"{cls_name} Lower CI")
    plt.scatter(wavelengths, upper_ci, color=color, marker="^", label=f"{cls_name} Upper CI")

# Combine extended and actual wavelengths for x-axis
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for the actual wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Customize y-axis to display two decimal places
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
plt.tick_params(axis="both", which="major", labelsize=12)
plt.tick_params(axis="both", which="minor", labelsize=12)

# Custom legend: Add single entry for CIs per class
handles, labels = plt.gca().get_legend_handles_labels()
ci_handles = [
    plt.Line2D([0], [0], linestyle="--", color=color, linewidth=1) for color in class_colors
]
ci_labels = [f"{cls_name} CI" for cls_name in class_labels.values()]
combined_handles = handles + ci_handles
combined_labels = labels + ci_labels
plt.legend(combined_handles, combined_labels, title="Classes:", loc="upper left", fontsize=12)

# Labels, title, and grid
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Reflectance", fontsize=14)
plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_shaded_ci_custom_legend.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")

# %%
#test 24 (works)
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_files[raster_file])
        mask_path = os.path.join(mask_dir, mask_files[mask_file])

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            bands_data[bands_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute prediction intervals
    interval = confidence_level * std_devs
    lower, upper = means - interval, means + interval

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
    )

    # Add markers for lower and upper bounds
    plt.scatter(wavelengths, lower_ci, color=color, marker="v", label=f"{cls_name} Lower Bound")
    plt.scatter(wavelengths, upper_ci, color=color, marker="^", label=f"{cls_name} Upper Bound")
    # plt.plot(
    #     wavelengths, lower_ci, linestyle="none", color=color, marker="_", markersize=10, label=f"{cls_name} Lower Bound"
    # )
    # plt.plot(
    #     wavelengths, upper_ci, linestyle="none", color=color, marker="_", markersize=10, label=f"{cls_name} Upper Bound"
    # )

# Combine extended and actual wavelengths for x-axis
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for the actual wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Customize y-axis to display two decimal places
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
plt.tick_params(axis="both", which="major", labelsize=12)
plt.tick_params(axis="both", which="minor", labelsize=12)

# Custom legend: Add CI with triangular symbols to legend
handles, labels = plt.gca().get_legend_handles_labels()
mean_handles = [
    plt.Line2D([0], [0], color=color, linewidth=2, label=f"{cls_name} mean")
    for color, cls_name in zip(class_colors, class_labels.values())
]
lower_bound_handles = [
    plt.Line2D([0], [0], color=color, marker="v", linestyle="None", label=f"{cls_name} Lower Bound")
    for color, cls_name in zip(class_colors, class_labels.values())
]
upper_bound_handles = [
    plt.Line2D([0], [0], color=color, marker="^", linestyle="None", label=f"{cls_name} upper/lower bound")
    for color, cls_name in zip(class_colors, class_labels.values())
]
combined_handles = mean_handles + upper_bound_handles
plt.legend(
    handles=combined_handles,
    title="Classes:",
    loc="upper left",
    fontsize=12,
    title_fontsize=14,
)
# Legend customization
# handles = [
#     plt.Line2D([0], [0], color=color, linewidth=2) for color in class_colors
# ]
# ci_handles = [
#     plt.Line2D([0], [0], color=color, marker="_", linestyle="None", markersize=10) for color in class_colors
# ]
# ci_labels = [f"{cls_name} CI" for cls_name in class_labels.values()]
# combined_handles = handles + ci_handles
# combined_labels = [f"{cls_name} Mean" for cls_name in class_labels.values()] + ci_labels

# plt.legend(
#     combined_handles,
#     combined_labels,
#     title="Classes:",
#     loc="upper left",
#     fontsize=12,
#     title_fontsize=14,
# )


# Labels, title, and grid
# plt.xlabel("Wavelength (nm)", fontsize=14)
# plt.ylabel("Reflectance", fontsize=14)
# plt.title("Mean Reflectance with 95% Prediction Intervals per Class", fontsize=16)
# plt.grid(axis="both", linestyle="--", alpha=0.5)

# Save and show plot
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_combined_ci_triangle_legend.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")


# %%
## test25
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_file)
        mask_path = os.path.join(mask_dir, mask_file)

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            # Replace NaN values
            bands_data[bands_data == src.nodata] = np.nan

            # Apply scale and offset
            # scale = src.scales[0] if src.scales else 1  # Default scale: 1
            # offset = src.offsets[0] if src.offsets else 0  # Default offset: 0
            # bands_data = bands_data * scale + offset

            # Ensure all values are non-negative
            # bands_data = np.clip(bands_data, 0, None)

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and prediction intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)

    means = np.nanmean(reflectance_data, axis=1)
    std_devs = np.nanstd(reflectance_data, axis=1)

    # Compute confidence intervals
    interval = confidence_level * std_devs
    lower = np.clip(means - interval, 0, None)  # Ensure lower CI is non-negative
    upper = means + interval


    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
    )

    # Add markers for lower and upper bounds
    plt.scatter(wavelengths, lower_ci, color=color, marker="v", label=f"{cls_name} Lower Bound")
    plt.scatter(wavelengths, upper_ci, color=color, marker="^", label=f"{cls_name} Upper Bound")

# Combine extended and actual wavelengths for x-axis
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for the actual wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Customize y-axis to display two decimal places
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
plt.tick_params(axis="both", which="major", labelsize=12)
plt.tick_params(axis="both", which="minor", labelsize=12)

# Custom legend: Add CI with triangular symbols to legend
handles, labels = plt.gca().get_legend_handles_labels()
mean_handles = [
    plt.Line2D([0], [0], color=color, linewidth=2, label=f"{cls_name} mean")
    for color, cls_name in zip(class_colors, class_labels.values())
]
lower_bound_handles = [
    plt.Line2D([0], [0], color='black', marker="v", linestyle="None", label=f"{cls_name} Lower Bound")
    for color, cls_name in zip(class_colors, class_labels.values())
]
upper_bound_handles = [
    plt.Line2D([0], [0], color=color, marker="^", linestyle="None", label=f"{cls_name} upper/lower bound")
    for color, cls_name in zip(class_colors, class_labels.values())
]
combined_handles = mean_handles + upper_bound_handles
plt.legend(
    handles=combined_handles,
    title="Classes:",
    loc="upper left",
    fontsize=12,
    title_fontsize=14,
)

# Labels, title, and grid
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_combined_ci_triangle_legend.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")


# %%
### test 26
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]

# Wavelengths
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range
confidence_level = 1.96  # For 95% confidence interval

# Aggregated reflectance data storage
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

# Process input files
for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f: f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f: f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files.values(), mask_files.values()), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_file)
        mask_path = os.path.join(mask_dir, mask_file)

        with rasterio.open(raster_path) as src:
            bands_data = src.read(out_dtype="float32")
            # Replace NaN values
            bands_data[bands_data == src.nodata] = np.nan

            # Apply scale and offset
            # scale = src.scales[0] if src.scales else 1  # Default scale: 1
            # offset = src.offsets[0] if src.offsets else 0  # Default offset: 0
            # bands_data = bands_data * scale + offset

            # # Ensure all values are non-negative
            # bands_data = np.clip(bands_data, 0, None)

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype("float32")
            mask[mask == src_mask.nodata] = np.nan

        valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

        for cls, cls_name in class_labels.items():
            class_mask = (mask == cls) & valid_pixels_mask
            if not np.any(class_mask):
                continue

            reflectance_values = bands_data[:, class_mask]
            aggregated_reflectance[cls_name].append(reflectance_values)

# Compute mean reflectance and confidence intervals
results = []

for cls_name, reflectance_list in aggregated_reflectance.items():
    if not reflectance_list:
        continue

    reflectance_data = np.hstack(reflectance_list)  # Combine data from all tiles
    n_samples = np.sum(~np.isnan(reflectance_data), axis=1)  # Count non-NaN pixels per band

    means = np.nanmean(reflectance_data, axis=1)  # Mean reflectance per band
    std_devs = np.nanstd(reflectance_data, axis=1)  # Standard deviation per band
    sem = std_devs / np.sqrt(n_samples)  # Standard error of the mean

    # Compute confidence intervals using SEM
    lower = np.clip(means - confidence_level * sem, 0, None)  # Ensure non-negative lower CI
    upper = means + confidence_level * sem

    # Store results for each class and wavelength
    for i, band in enumerate(bands):
        results.append({
            "Class": cls_name,
            "Wavelength": band,
            "Mean Reflectance": means[i],
            "Lower CI": lower[i],
            "Upper CI": upper[i],
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Class", "Wavelength"])

# Display the results as a table
print(results_df)

# Save the results to a CSV file
results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved results table to {results_csv_path}")

# Plotting
plt.figure(figsize=(14, 8))

for cls_name, color in zip(class_labels.values(), class_colors):
    class_data = results_df[results_df["Class"] == cls_name]
    wavelengths = class_data["Wavelength"]
    means = class_data["Mean Reflectance"]
    lower_ci = class_data["Lower CI"]
    upper_ci = class_data["Upper CI"]

    # Plot mean reflectance as a full line
    plt.plot(
        wavelengths,
        means,
        label=f"{cls_name} Mean",
        color=color,
        linewidth=2,
    )

    # Plot confidence intervals as shaded areas
    plt.fill_between(
        wavelengths,
        lower_ci,
        upper_ci,
        color=color,
        alpha=0.2,
    )

    # Add markers for lower and upper bounds
    plt.scatter(wavelengths, lower_ci, color=color, marker="v", label=f"{cls_name} Lower Bound")
    plt.scatter(wavelengths, upper_ci, color=color, marker="^", label=f"{cls_name} Upper Bound")

# Combine extended and actual wavelengths for x-axis
xticks_combined = sorted(set(x_axis_wavelengths + bands))
plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

# Draw vertical gridlines for the actual wavelengths
for x in bands:
    plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

# Customize y-axis to display two decimal places
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
plt.tick_params(axis="both", which="major", labelsize=12)
plt.tick_params(axis="both", which="minor", labelsize=12)

# Custom legend: Add CI with triangular symbols to legend
handles, labels = plt.gca().get_legend_handles_labels()
mean_handles = [
    plt.Line2D([0], [0], color=color, linewidth=2, label=f"{cls_name} mean")
    for color, cls_name in zip(class_colors, class_labels.values())
]
lower_bound_handles = [
    plt.Line2D([0], [0], color=color, marker="v", linestyle="None", label=f"{cls_name} Lower Bound")
    for color, cls_name in zip(class_colors, class_labels.values())
]
upper_bound_handles = [
    plt.Line2D([0], [0], color=color, marker="^", linestyle="None", label=f"{cls_name} upper/lower bound")
    for color, cls_name in zip(class_colors, class_labels.values())
]
combined_handles = mean_handles + upper_bound_handles
plt.legend(
    handles=combined_handles,
    title="Classes:",
    loc="upper left",
    fontsize=12,
    title_fontsize=14,
)

# Labels, title, and grid
plt.tight_layout()
output_plot_path = os.path.join(output_dir, "mean_reflectance_with_combined_ci_triangle_legend.png")
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nSaved plot to {output_plot_path}")


# %%
##test 27
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and colors
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]
confidence_level = 1.96  # For 95% confidence interval

# Wavelengths
bands = [475, 560, 668, 717, 842]  # Actual wavelengths for the bands
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]  # Extended x-axis range for ticks

# Functions
def extract_band_statistics(src):
    """Extract statistics for each band."""
    band_stats = {}
    for i in range(src.count):
        band_num = i + 1
        band_data = src.read(band_num)
        band_stats[band_num] = {
            "NoData": src.nodatavals[i],  # Extract No-Data value from metadata
            "Min": np.nanmin(band_data),  # Minimum value in the band
            "Max": np.nanmax(band_data),  # Maximum value in the band
        }
    return band_stats


def process_file(raster_path, mask_path):
    """Process a single raster and mask file."""
    results = []
    with rasterio.open(raster_path) as src:
        bands_data = src.read(out_dtype="float32")
        bands_data[bands_data == src.nodata] = np.nan  # Replace NaN values
        band_stats = extract_band_statistics(src)  # Get band statistics

    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1).astype("float32")
        mask[mask == src_mask.nodata] = np.nan

    if bands_data.shape[1:] != mask.shape:
        print(f"Shape mismatch: Raster {bands_data.shape[1:]} vs Mask {mask.shape}")
        return results

    valid_pixels_mask = ~np.isnan(mask) & np.all(~np.isnan(bands_data), axis=0)

    for cls, cls_name in class_labels.items():
        class_mask = (mask == cls) & valid_pixels_mask
        if not np.any(class_mask):
            continue
        reflectance_values = bands_data[:, class_mask]
        results.append((cls_name, reflectance_values, band_stats))

    return results

# def compute_statistics(aggregated_reflectance):
#     """Compute mean reflectance and confidence intervals."""
#     results = []
#     for cls_name, reflectance_list in aggregated_reflectance.items():
#         if not reflectance_list:
#             continue

#         reflectance_data = np.hstack([x[0] for x in reflectance_list])  # Combine data
#         band_stats = reflectance_list[0][1]  # Use band stats from the first file

#         n_samples = np.sum(~np.isnan(reflectance_data), axis=1)
#         means = np.nanmean(reflectance_data, axis=1)
#         std_devs = np.nanstd(reflectance_data, axis=1)
#         sem = std_devs / np.sqrt(n_samples)

#         lower = np.clip(means - confidence_level * sem, 0, None)
#         upper = means + confidence_level * sem

#         for i, band_index in enumerate(band_stats.keys()):
#             results.append({
#                 "Class": cls_name,
#                 "Band": band_index,
#                 "NoData": band_stats[band_index]["NoData"],  # Include No-Data value
#                 "Min": band_stats[band_index]["Min"],        # Minimum value
#                 "Max": band_stats[band_index]["Max"],        # Maximum value
#                 "Mean Reflectance": means[i],
#                 "Lower CI": lower[i],
#                 "Upper CI": upper[i],
#             })

#     return pd.DataFrame(results)
def compute_statistics(aggregated_reflectance):
    """Compute mean reflectance and confidence intervals using combined band stats."""
    results = []

    for cls_name, reflectance_list in aggregated_reflectance.items():
        if not reflectance_list:
            continue

        # Combine reflectance data across all files for the class
        reflectance_data = np.hstack([x[0] for x in reflectance_list])

        # Aggregate band stats across all files
        combined_band_stats = {}
        all_band_stats = [x[1] for x in reflectance_list]  # Extract band stats from all files
        for band_num in all_band_stats[0].keys():  # Assume all files have the same band structure
            combined_band_stats[band_num] = {
                "NoData": all_band_stats[0][band_num]["NoData"],  # Assuming NoData is consistent
                "Min": min(stats[band_num]["Min"] for stats in all_band_stats),
                "Max": max(stats[band_num]["Max"] for stats in all_band_stats),
            }

        # Compute mean reflectance and confidence intervals
        n_samples = np.sum(~np.isnan(reflectance_data), axis=1)  # Count valid pixels for each band
        means = np.nanmean(reflectance_data, axis=1)  # Mean reflectance per band
        std_devs = np.nanstd(reflectance_data, axis=1)  # Standard deviation per band
        sem = std_devs / np.sqrt(n_samples)  # Standard error of the mean

        lower = np.clip(means - confidence_level * sem, 0, None)  # Lower bound of 95% CI
        upper = means + confidence_level * sem  # Upper bound of 95% CI

        # Store results for each band
        for i, band_num in enumerate(combined_band_stats.keys()):
            results.append({
                "Class": cls_name,
                "Band": band_num,
                "NoData": combined_band_stats[band_num]["NoData"],
                "Min": combined_band_stats[band_num]["Min"],
                "Max": combined_band_stats[band_num]["Max"],
                "Mean Reflectance": means[i],
                "Lower CI": lower[i],
                "Upper CI": upper[i],
            })

    return pd.DataFrame(results)



def plot_results_with_bounds(results_df, output_path):
    """Plot reflectance statistics with confidence intervals and bounds."""
    plt.figure(figsize=(14, 8))

    for cls_name, color in zip(class_labels.values(), class_colors):
        class_data = results_df[results_df["Class"] == cls_name]
        wavelengths = [bands[int(band) - 1] for band in class_data["Band"]]  # Map bands to wavelengths
        means = class_data["Mean Reflectance"]
        lower_ci = class_data["Lower CI"]
        upper_ci = class_data["Upper CI"]

        # Plot mean reflectance as a full line
        plt.plot(
            wavelengths,
            means,
            label=f"{cls_name} Mean",
            color=color,
            linewidth=2,
        )

        # Plot confidence intervals as shaded areas
        plt.fill_between(
            wavelengths,
            lower_ci,
            upper_ci,
            color=color,
            alpha=0.2,
        )

        # Add markers for lower and upper bounds
        plt.scatter(
            wavelengths,
            lower_ci,
            color=color,
            edgecolor="black",
            marker="v",  # Downward triangle
            label=f"{cls_name} Lower Bound",
        )
        plt.scatter(
            wavelengths,
            upper_ci,
            color=color,
            edgecolor="black",
            marker="^",  # Upward triangle
            label=f"{cls_name} Upper Bound",
        )

    # Combine extended and actual wavelengths for x-axis
    xticks_combined = sorted(set(x_axis_wavelengths + bands))
    plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

    # Draw vertical gridlines for the actual wavelengths
    for x in bands:
        plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

    # Customize y-axis
    y_max = results_df["Upper CI"].max()
    y_ticks = np.arange(0, y_max + 0.03, 0.03)
    plt.ylim(0, y_ticks[-1])
    plt.yticks(y_ticks)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tick_params(axis="both", which="minor", labelsize=14)

    # X and Y Labels, Title
    # plt.xlabel("Wavelength (nm)", fontsize=14)
    # plt.ylabel("Reflectance", fontsize=14)
    # plt.title("Mean Reflectance with Confidence Intervals per Class", fontsize=16)

    # Grid lines
    # plt.grid(axis="both", linestyle="--", alpha=0.5)

    # Custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    mean_handles = [
        plt.Line2D([0], [0], color=color, linewidth=2, label=f"{cls_name} mean")
        for color, cls_name in zip(class_colors, class_labels.values())
    ]
    lower_bound_handles = [
        plt.Line2D([0], [0], color=color, marker="v", linestyle="None", markersize=8, markeredgecolor="black", label=f"{cls_name} Lower Bound")
        for color, cls_name in zip(class_colors, class_labels.values())
    ]
    upper_bound_handles = [
        plt.Line2D([0], [0], color=color, marker="^", linestyle="None", markersize=8, markeredgecolor="black", label=f"{cls_name} lower/upper bound")
        for color, cls_name in zip(class_colors, class_labels.values())
    ]
    combined_handles = mean_handles + upper_bound_handles
    plt.legend(
        handles=combined_handles,
        title="Classes:",
        loc="upper left",
        fontsize=13,
        title_fontsize=14,
    )

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()




# Main Script
aggregated_reflectance = {cls_name: [] for cls_name in class_labels.values()}

for dirs in input_dirs:
    raster_dir = dirs["raster_dir"]
    mask_dir = dirs["mask_dir"]

    raster_files = {f for f in os.listdir(raster_dir) if f.endswith(".tif")}
    mask_files = {f for f in os.listdir(mask_dir) if f.endswith(".tif")}

    for raster_file, mask_file in tqdm(zip(raster_files, mask_files), desc=f"Processing {raster_dir}"):
        raster_path = os.path.join(raster_dir, raster_file)
        mask_path = os.path.join(mask_dir, mask_file)

        file_results = process_file(raster_path, mask_path)
        for cls_name, reflectance_values, band_stats in file_results:
            aggregated_reflectance[cls_name].append((reflectance_values, band_stats))

results_df = compute_statistics(aggregated_reflectance)

if not results_df.empty:
    print("\nReflectance Statistics with Confidence Intervals:")
    print(results_df)

    # Define the CSV output path
    results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci.csv")
    
    # Add confidence interval values (Lower CI and Upper CI) to the CSV
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\nSaved results table with confidence intervals to {results_csv_path}")

    # Plot and save the results
    plot_results_with_bounds(results_df, os.path.join(output_dir, "reflectance_plot_sites.png"))
else:
    print("No valid data was processed.")
    
    
# %%
##test 28
import os
import numpy as np
import pandas as pd
import rasterio
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Input directories
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

# Output directory
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined"
os.makedirs(output_dir, exist_ok=True)

# Class labels and confidence level
class_labels = {0: "BE", 1: "NPV", 2: "PV", 3: "SI", 4: "WI"}
class_colors = ["#dae22f", "#6332ea", "#e346ee", "#6da4d4", "#68e8d3"]
confidence_level = 1.96  # For 95% confidence interval
bands = [475, 560, 668, 717, 842]  # Actual wavelengths
x_axis_wavelengths = [400, 500, 600, 700, 800, 900]

# Functions
@delayed
def process_file(raster_path, mask_path):
    """Process raster and mask files to extract data per class."""
    try:
        with rasterio.open(raster_path) as src:
            raster_data = src.read(out_dtype="float32")
            raster_data[raster_data == src.nodata] = np.nan

        with rasterio.open(mask_path) as src_mask:
            mask_data = src_mask.read(1).astype("float32")
            mask_data[mask_data == src_mask.nodata] = np.nan

        return raster_data, mask_data
    except Exception as e:
        print(f"Error processing files: {raster_path}, {mask_path}. Error: {e}")
        return None, None

def compute_statistics(aggregated_reflectance):
    """Compute statistics for all classes and bands."""
    results = []
    for cls_name, class_data in aggregated_reflectance.items():
        for band_idx, band_values in class_data.items():
            if len(band_values) > 0:
                band_values = np.hstack(band_values)
                n = len(band_values)
                mean = np.nanmean(band_values)
                min_val = np.nanmin(band_values)
                max_val = np.nanmax(band_values)
                std_dev = np.nanstd(band_values)
                sem = std_dev / np.sqrt(n) if n > 0 else 0
                lower_ci = max(mean - confidence_level * sem, 0)
                upper_ci = mean + confidence_level * sem

                results.append({
                    "Class": cls_name,
                    "Band": bands[band_idx],
                    "Count": n,
                    "Mean Reflectance": mean,
                    "Min": min_val,
                    "Max": max_val,
                    "Lower CI": lower_ci,
                    "Upper CI": upper_ci,
                })
    return pd.DataFrame(results)

def plot_results_with_bounds(results_df, output_path):
    """Plot reflectance statistics with confidence intervals and bounds."""
    plt.figure(figsize=(14, 8))

    for cls_name, color in zip(class_labels.values(), class_colors):
        class_data = results_df[results_df["Class"] == cls_name]
        wavelengths = class_data["Band"]
        means = class_data["Mean Reflectance"]
        lower_ci = class_data["Lower CI"]
        upper_ci = class_data["Upper CI"]

        # Plot mean reflectance as a full line
        plt.plot(
            wavelengths,
            means,
            label=f"{cls_name} Mean",
            color=color,
            linewidth=2,
        )

        # Plot confidence intervals as shaded areas
        plt.fill_between(
            wavelengths,
            lower_ci,
            upper_ci,
            color=color,
            alpha=0.2,
        )

        # Add markers for lower and upper bounds
        plt.scatter(
            wavelengths,
            lower_ci,
            color=color,
            edgecolor="black",
            marker="v",  # Downward triangle
            label=f"{cls_name} Lower Bound",
        )
        plt.scatter(
            wavelengths,
            upper_ci,
            color=color,
            edgecolor="black",
            marker="^",  # Upward triangle
            label=f"{cls_name} Upper Bound",
        )

    # Combine extended and actual wavelengths for x-axis
    xticks_combined = sorted(set(x_axis_wavelengths + bands))
    plt.xticks(xticks_combined, labels=[f"{wl}" for wl in xticks_combined])

    # Draw vertical gridlines for the actual wavelengths
    for x in bands:
        plt.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

    # Customize y-axis
    y_max = results_df["Upper CI"].max()
    y_ticks = np.arange(0, y_max + 0.03, 0.03)
    plt.ylim(0, y_ticks[-1])
    plt.yticks(y_ticks)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tick_params(axis="both", which="minor", labelsize=14)

    # Custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    mean_handles = [
        plt.Line2D([0], [0], color=color, linewidth=2, label=f"{cls_name} mean")
        for color, cls_name in zip(class_colors, class_labels.values())
    ]
    lower_bound_handles = [
        plt.Line2D([0], [0], color=color, marker="v", linestyle="None", markersize=8, markeredgecolor="black", label=f"{cls_name} Lower Bound")
        for color, cls_name in zip(class_colors, class_labels.values())
    ]
    upper_bound_handles = [
        plt.Line2D([0], [0], color=color, marker="^", linestyle="None", markersize=8, markeredgecolor="black", label=f"{cls_name} lower/upper bound")
        for color, cls_name in zip(class_colors, class_labels.values())
    ]
    combined_handles = mean_handles + upper_bound_handles
    plt.legend(
        handles=combined_handles,
        title="Classes:",
        loc="upper left",
        fontsize=13,
        title_fontsize=14,
    )

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# Main processing
aggregated_reflectance = {cls_name: {i: [] for i in range(len(bands))} for cls_name in class_labels.values()}
tasks = []

for dirs in input_dirs:
    raster_dir, mask_dir = dirs["raster_dir"], dirs["mask_dir"]
    raster_files = sorted([f for f in os.listdir(raster_dir) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

    for raster_file, mask_file in zip(raster_files, mask_files):
        raster_path, mask_path = os.path.join(raster_dir, raster_file), os.path.join(mask_dir, mask_file)
        tasks.append(process_file(raster_path, mask_path))

# Compute all delayed tasks
with ProgressBar():
    processed_files = compute(*tasks)

# Populate aggregated reflectance
for raster_data, mask_data in processed_files:
    if raster_data is not None and mask_data is not None:
        for class_value, class_name in class_labels.items():
            class_mask = (mask_data == class_value)
            for band_idx in range(len(bands)):
                band_values = raster_data[band_idx, class_mask]
                if band_values.size > 0:
                    aggregated_reflectance[class_name][band_idx].append(band_values)

# Compute statistics and save results
results_df = compute_statistics(aggregated_reflectance)
if not results_df.empty:
    results_csv_path = os.path.join(output_dir, "reflectance_statistics_with_ci_sites.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved results to {results_csv_path}")

    # Plot results
    plot_path = os.path.join(output_dir, "reflectance_plot_sites.png")
    plot_results_with_bounds(results_df, plot_path)
else:
    print("No valid data found for processing.")

# %%
