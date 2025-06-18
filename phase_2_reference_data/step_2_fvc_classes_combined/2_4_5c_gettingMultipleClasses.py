#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:18:54 2023
@author: lauransotomayor
"""

####### concatenate shp 
#%%
'''multi-class mask, each class is represented by a unique integer value.'''
import tifffile as tiff
import numpy as np
import rasterio
from tifffile import imread as tiff_imread

############# Creates a combined multi-class single-band GeoTIFF mask as output 
# ONLY ONE FILE/TILE PROCESSING
# mask_sam_be_npv ='/media/laura/Extreme SSD/qgis/calperumResearch/sampleData/site1_supersite_DD0001/masks/masks_fvc/masks_sam/masks_sam_v0/mask_semanticSegmentation_1874.tif'
# mask_sam_be_npv ='/media/laura/Extreme SSD/qgis/calperumResearch/sampleData/site1_supersite_DD0001/masks/masks_fvc/masks_sam/masks_sam_v1/mask_semanticSegmentation_otsu_1874.tif'
# mask_bbox_pv ='/media/laura/Extreme SSD/qgis/calperumResearch/sampleData/site1_supersite_DD0001/masks/masks_fvc/masks_pv/mask_clipped_1870.tif'

# Paths to your input masks and desired output file
mask_be_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/mask_be_globStat_percentile_tiles_rgb_3072.16.tif'
mask_pv_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/mask_pv_tiles_rgb_3072.16.tif'
output_file = 'masks_combined_fvc_3c_3072.16.tif'  # Adjust the path accordingly

# Read the BE and PV masks
mask_be = tiff_imread(mask_be_path)
mask_pv = tiff_imread(mask_pv_path)

# Ensure masks are binary (0 or 1)
mask_be_clipped = np.clip(mask_be, 0, 1)
mask_pv_clipped = np.clip(mask_pv, 0, 1)

# Initialize the combined mask as 0 for NPV by default
combined_mask = np.zeros_like(mask_be_clipped, dtype=np.float32)

# Assign 1 to areas identified as BE (mask_be_clipped == 1)
combined_mask[mask_be_clipped == 1] = 0
combined_mask[mask_be_clipped == 0] = 1

# Assign 2 to areas identified as PV (mask_pv_clipped == 1)
combined_mask[mask_pv_clipped == 1] = 2

# Write the combined mask to a new file, using metadata from the BE mask for consistency
with rasterio.open(mask_be_path) as src:
    meta = src.meta.copy()

meta.update(dtype=rasterio.float32, count=1)  # Update metadata for single band and float32 data type

with rasterio.open(output_file, 'w', **meta) as dst:
    dst.write(combined_mask, 1)
    
#%%
########################## SAME CODE FOR MULTIPLE FILES PROCESSING
import os
import tifffile as tiff
import numpy as np
import rasterio

# Define the input folders containing your mask files
# mask_sam_be_npv ='/media/laura/Extreme SSD/qgis/calperumResearch/sampleData/site1_supersite_DD0001/masks/masks_fvc/masks_sam/masks_sam_v0/'
# mask_sam_be_npv ='/media/laura/Extreme SSD/qgis/calperumResearch/sampleData/site1_supersite_DD0001/masks/masks_fvc/masks_sam/masks_sam_v1/'
# mask_bbox_pv ='/media/laura/Extreme SSD/qgis/calperumResearch/sampleData/site1_supersite_DD0001/masks/masks_fvc/masks_pv/'

#site1
be_dir = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/'
# mask_npv = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/glob_stats/npv_class'
pv_dir = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/'

# Define the output folder for saving the combined masks
output_dir = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/fvc_class/'


# List all files in the BE and PV directories
be_files = os.listdir(be_dir)
pv_files = os.listdir(pv_dir)

# Function to extract a common identifier from filenames
def extract_common_identifier(filename):
    # This function should extract the part of the filename that matches between BE and PV files
    # For example, if the filename structure is "mask_<type>_tiles_rgb_3072.16.tif", extract the trailing identifier
    return filename.split('_')[-1]

# Mapping of PV files to their identifiers for quick lookup
pv_mapping = {extract_common_identifier(pv_file): pv_file for pv_file in pv_files}

# Process each BE file
for be_file in be_files:
    identifier = extract_common_identifier(be_file)
    
    # Check if there is a matching PV file
    if identifier in pv_mapping:
        be_path = os.path.join(be_dir, be_file)
        pv_path = os.path.join(pv_dir, pv_mapping[identifier])
        output_file = os.path.join(output_dir, f'masks_fvc_{identifier}')
        
        # Read the BE and PV masks
        mask_be = tiff_imread(be_path)
        mask_pv = tiff_imread(pv_path)

        # Ensure masks are binary (0 or 1)
        mask_be_clipped = np.clip(mask_be, 0, 1)
        mask_pv_clipped = np.clip(mask_pv, 0, 1)

        # Initialize the combined mask as 0 for NPV by default
        combined_mask = np.zeros_like(mask_be_clipped, dtype=np.float32)

        # Assign 1 to areas identified as BE (mask_be_clipped == 1)
        combined_mask[mask_be_clipped == 1] = 0
        combined_mask[mask_be_clipped == 0] = 1

        # Assign 2 to areas identified as PV (mask_pv_clipped == 1)
        combined_mask[mask_pv_clipped == 1] = 2

        # Write the combined mask to a new file, using metadata from the BE mask for consistency
        with rasterio.open(be_path) as src:
            meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1)  # Update metadata for single band and float32 data type

        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(combined_mask, 1)
        print(f"Processed and saved: {output_file}")
    else:
        print(f"No matching PV file for BE file: {be_file}")


#%%
import os
import numpy as np
import rasterio
from tifffile import imread as tiff_imread
import geopandas as gpd
from rasterio.features import shapes
import shapely.geometry as geometry

# Your directory paths and file processing code remains the same until after you've created 'combined_mask'

# Convert the raster to polygons
mask_data = combined_mask
with rasterio.open(be_path) as src:
    transform = src.transform

mask_polygons = []
values = []

for shape, value in shapes(mask_data, mask=mask_data > 0, transform=transform):
    if value > 0:  # You might adjust this if you want to include 0 values as well
        mask_polygons.append(geometry.shape(shape))
        values.append(value)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame({'vegetation_class': values, 'geometry': mask_polygons})

# Set the CRS (Coordinate Reference System) to match your raster data
# with rasterio.open(be_path) as src:
#     gdf.crs = src.crs

# # Save to Shapefile
# output_shapefile = os.path.join(output_dir, f'masks_fvc_{identifier}.shp')
# gdf.to_file(output_shapefile)

print(f"Shapefile saved to: {output_shapefile}")
          
            
# %%
############# display
import os
import numpy as np
import rasterio
import rasterio.transform
from tifffile import imread  # Assuming tiff.imread refers to tifffile.imread
import matplotlib.pyplot as plt

def plot_and_save_combined_mask(combined_mask, output_dir):
    """
    Plots the combined mask and saves the figure to a file.

    Parameters:
    - combined_mask: The numpy array of the combined mask.
    - output_file: The path to save the output plot.
    """
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis  # Colormap
    # Display the image
    cax = ax.imshow(combined_mask, cmap=cmap, interpolation='nearest')
    ax.set_title('Combined Mask with 3 Classes')

    # Add colorbar to match with classes
    cbar = fig.colorbar(cax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Bare earth class', 'NPV class', 'PV class'])  # Vertically oriented colorbar

    # Save the figure
    plt.savefig(output_dir)
    plt.close(fig)  # Close the plot to free memory
    

# Example call to the plot function, adjust paths as necessary
# combined_mask is your generated mask array
custom_plot_filename = f'mask_fvc_3c_{identifier}_plot.png'
plot_and_save_combined_mask(combined_mask, custom_plot_filename)


# %%
########### Raster to shp
import geopandas as gpd
import rasterio
from rasterio.features import shapes
import numpy as np
from shapely.geometry import shape

# Specify your raster path and the desired output shapefile path
input_tiff  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/fvc_class/masks_fvc_3072.16.tif'
output_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/fvc_class/annotation_shp/masks_fvc_3072.16.shp'


# Ensure this function processes the raster data as uint8
def raster_to_shapefile(input_raster, output_shapefile, class_value):
    with rasterio.open(input_raster) as src:
        # Convert the raster data to uint8
        image = src.read(1).astype(np.uint8)  # Ensure image is uint8
        meta = src.meta

        # Update the meta to reflect uint8 data type
        meta.update(dtype=np.uint8)

        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(image, mask=(image == class_value), transform=src.transform))
            if v == class_value
        )
        
        geoms = list(results)
        if geoms:  # Check if there are any geometries to prevent empty GeoDataFrame creation
            gdf = gpd.GeoDataFrame.from_features(geoms)
            gdf.crs = src.crs
            gdf.to_file(output_shapefile)

# Process for each class value (0, 1, 2) and save to separate shapefiles
for class_value in [0, 1, 2]:
    class_output_shapefile = output_shapefile.replace('.shp', f'_class_{class_value}.shp')
    raster_to_shapefile(input_tiff, class_output_shapefile, class_value)
    print(f"Shapefile for class {class_value} saved to: {class_output_shapefile}")



#%%
########### Raster to shp 2
import geopandas as gpd
import rasterio
from rasterio.features import shapes
import numpy as np
from shapely.geometry import shape

# Specify your raster path and the desired output shapefile path
input_tiff  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/with_id/masks_fvc_3072.16.tif'
output_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/fvc_class/annotation_shp/masks_fvc_3072.16.shp'

# Class names mapping
class_names = {
    0: "nan",  # Bare Earth
    1: "be",  # Bare Earth
    2: "npv", # Non-Photosynthetic Vegetation
    3: "pv",   # Photosynthetic Vegetation
    4: "si",
    5: "wi"
}

# Initialize an empty list for accumulating geometries and their properties
geometries = []
# Initialize a counter for unique IDs
feature_id = 0

with rasterio.open(input_tiff) as src:
    image = src.read(1)  # Reading the first (and only) band
    transform = src.transform
    
    # Iterate over unique class values present in the raster
    for class_value in np.unique(image):
        # Skip if class_value not defined in class_names (e.g., if there's an unexpected value)
        if class_value not in class_names:
            continue
        
        # Creating a binary mask for the current class
        mask = image == class_value
        # Vectorize shapes where the mask is True
        for geom, value in shapes(image, mask=mask, transform=transform):
            # Check to ensure the value matches the class_value
            if value == class_value:
                geom_shape = shape(geom)
                centroid = geom_shape.centroid
                
                # Append a dictionary with the geometry, class name, centroid coordinates, and a unique ID
                geometries.append({
                    'geometry': geom_shape,
                    'properties': {
                        'id': feature_id,
                        'class': class_names[class_value],
                        'x': centroid.x,  # Longitude
                        'y': centroid.y   # Latitude
                        
                    }
                })
                feature_id += 1  # Increment the ID for the next feature

# Create a GeoDataFrame from the accumulated geometries and properties
gdf = gpd.GeoDataFrame.from_features(geometries, crs=src.crs)

# Save the GeoDataFrame as a shapefile
gdf.to_file(output_shapefile)

print(f"Shapefile with class polygons, centroids, and IDs saved to: {output_shapefile}")

# %%
########## plot
import geopandas as gpd
import matplotlib.pyplot as plt

# Assuming shapefile_path is the directory where the shapefile is saved
# and output_filename is the name of the shapefile created from JSON annotations
shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/fvc_class/annotation_shp/'
output_filename= 'masks_fvc_3072.16.shp'

# output_filename = 'globStat_percentile_tiles_rgb_fobs.125_poly_annotated.shp'
shapefile_full_path = os.path.join(shapefile_path, output_filename)

# Read the shapefile
gdf = gpd.read_file(shapefile_full_path)

# Plot the shapefile
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size as needed
gdf.plot(ax=ax)

# Setting plot titles and labels, if needed
ax.set_title('Shapefile Plot')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()


# %%
import geopandas as gpd


# shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/glob_stats/tiles_rgb_large_2048/pv_class/globStat_percentile_tiles_rgb_fobs.125_poly_annotated.shp'
shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/fvc_class/annotation_shp/'


annotated_shp = gpd.read_file(shapefile_path)
annotated_shp.head()

# %%
##### fvc classes
import rasterio
import matplotlib.pyplot as plt

def plot_raster(raster_path, save_path):
    """
    Reads a raster file and plots it with a color bar matched to specific classes.

    Parameters:
    - raster_path: The file path to the raster file.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band
        data = src.read(1)
    
         # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data, cmap='viridis', interpolation='nearest')
        ax.set_title('FVC classes mask')
        ax.axis('off')  # Turn off axis numbers and ticks
        
        # Create colorbar to match the image height
        colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        # Set the ticks for the colorbar
        colorbar.set_ticks([0, 1, 2, 3])  # Make sure these values correspond to your data
        # Set the tick labels for the colorbar
        colorbar.ax.set_yticklabels(['Bare earth class', 'NPV class', 'PV class', 'SI class'])  # Customize tick labels as needed

        # Adjust layout to handle subplots nicely
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust the spacing between plots if there are multiple subplots

        # plt.show()
        
         # Save the figure to a file
        plt.savefig(save_path)
        plt.close(fig)  # Close the plot to free up memory

        print(f"Image saved as {save_path}")

# Example call to the function
raster_file_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/preprocess/mask_fvc_3072.16.tif'
output_image_path = '/home/laura/Documents/code/ecosystem_composition/phase_2_reference_data/3_pv_npv_be_classes/fvc_mask.png'

plot_raster(raster_file_path, output_image_path)


#%%
### improve fvc
import rasterio
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.colors import ListedColormap

def plot_raster(raster_path, rgb_path, save_path):
    """
    Reads a raster file and an RGB image file, displays them side-by-side with a customized color bar for the raster,
    and optionally saves the output image. The raster visualization is intended for visualization of classes such as 'Bare Earth'.

    Parameters:
    - raster_path: The file path to the raster file.
    - rgb_path: The file path to the RGB image file.
    - save_path: The file path where the image should be saved.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band
        data = src.read(1)
    
    # Read the RGB image
    rgb_image = imread(rgb_path)
    
    # Define a discrete color palette for the raster classes
    class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4']  # Colors for Bare earth, NPV, PV, SI classes (medium)
    cmap = ListedColormap(class_colors)
    
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    # Display RGB image
    axes[0].imshow(rgb_image)
    # axes[0].set_title('False Colour Composite - Multispectral Imagery')
    axes[0].axis('off')  # Turn off axis numbers and ticks

    # Display Binary Mask
    im = axes[1].imshow(data, interpolation='nearest', cmap=cmap)
    # axes[1].set_title('FVC Class Mask')
    axes[1].axis('off')  # Turn off axis numbers and ticks
    
    # Create a color bar to match the image height
    colorbar = fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    # colorbar.set_ticks([0, 1, 2, 3])  # Set ticks to match the number of classes
    # colorbar.ax.set_yticklabels(['Bare earth class', 'NPV class', 'PV class', 'SI class'], fontsize=16)  # Customize tick labels as needed
    colorbar.set_ticks([0, 1, 2, 3])
    colorbar.ax.set_yticklabels(['BE class', 'NPV class', 'PV class', 'SI class'], fontsize=16)  # Customize tick labels as needed

    # Adjust layout to handle subplots nicely
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.1)  # Adjust the spacing between plots

    # Display plot
    # plt.show()
    
    # Save the figure to a file
    plt.savefig(save_path)
    # plt.close(fig)  # Close the plot to free up memory

    print(f"Image saved as {save_path}")

# Example call to the function
# rgb_file_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_1_DD0001/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.16.tif'
# raster_file_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/preprocess/mask_fvc_3072.16.tif'
# output_image_path = '/home/laura/Documents/code/ecosystem_composition/phase_2_reference_data/3_pv_npv_be_classes/fvc_mask.png'

rgb_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_256/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral_2620.tif'
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_256/raw/fvc_class/mask_fvc/mask_tiles_multispectral_2620.tif'
output_image_path = '/home/laura/Documents/code/ecosystem_composition/phase_2_reference_data/3_pv_npv_be_classes/fvc_mask_256.png'

plot_raster(raster_file_path, rgb_file_path, output_image_path)


#%%
# FOR 256PX
import rasterio
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.colors import ListedColormap

def plot_raster(raster_path, rgb_path, save_rgb_path, save_raster_path):
    """
    Reads a raster file and an RGB image file, displays them side-by-side with a customized color bar for the raster,
    and saves the output images separately. The raster visualization is intended for visualization of classes such as 'Bare Earth'.

    Parameters:
    - raster_path: The file path to the raster file.
    - rgb_path: The file path to the RGB image file.
    - save_rgb_path: The file path where the RGB image should be saved.
    - save_raster_path: The file path where the raster image should be saved.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band
        data = src.read(1)
    
    # Read the RGB image
    rgb_image = imread(rgb_path)
    
    # Define a discrete color palette for the raster classes
    class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4']  # Colors for Bare earth, NPV, PV, SI classes (medium)
    cmap = ListedColormap(class_colors)
    
    # Plotting RGB image
    fig_rgb, ax_rgb = plt.subplots(figsize=(10, 10))
    ax_rgb.imshow(rgb_image)
    ax_rgb.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(save_rgb_path, dpi=300, bbox_inches='tight')
    plt.close(fig_rgb)  # Close the plot to free up memory

    # Plotting Raster Mask
    fig_raster, ax_raster = plt.subplots(figsize=(10, 10))
    im = ax_raster.imshow(data, interpolation='nearest', cmap=cmap)
    ax_raster.axis('off')  # Turn off axis numbers and ticks
    colorbar = fig_raster.colorbar(im, ax=ax_raster, orientation='vertical', fraction=0.046, pad=0.04)
    colorbar.set_ticks([0, 1, 2, 3])
    colorbar.ax.set_yticklabels(['BE class', 'NPV class', 'PV class', 'SI class'], fontsize=24)  # Customize tick labels as needed
    plt.savefig(save_raster_path, dpi=300, bbox_inches='tight')
    plt.close(fig_raster)  # Close the plot to free up memory

    print(f"RGB image saved as {save_rgb_path}")
    print(f"Raster image saved as {save_raster_path}")

# Example call to the function
# rgb_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_256/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral_2620.tif'
# raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/mask_fvc_subsample/mask_tiles_multispectral_2620.tif'
# raster_file_path = '/media/laura/Extreme SSD/code/ecosystem_composition/phase_3_models/unet_model/inference/medium/1024/120ep/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55/inference_tile22/tile_37_prediction.tif'
raster_file_path = '/media/laura/Extreme SSD/code/ecosystem_composition/phase_3_models/unet_model/inference/medium/1024/120ep/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_117/inference_tile22/tile_37_prediction.tif'
save_rgb_path = '/media/laura/Extreme SSD/code/ecosystem_composition/phase_2_reference_data/step_2_fvc_classes_combined/fvc_predictor_256.png'
save_raster_path = '/media/laura/Extreme SSD/code/ecosystem_composition/phase_2_reference_data/step_2_fvc_classes_combined/fvc_mask_inference117_256.png'

plot_raster(raster_file_path, rgb_file_path, save_rgb_path, save_raster_path)



# %%
### 3 images
import rasterio
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.colors import ListedColormap


def plot_raster(raster_path, rgb_path1, rgb_path2, save_path):
    """
    Reads a raster file and two RGB image files, displays them side-by-side with a customized color bar for the raster,
    and optionally saves the output image. The raster visualization is intended for visualization of classes such as 'Bare Earth'.

    Parameters:
    - raster_path: The file path to the raster file.
    - rgb_path1: The first RGB image file path.
    - rgb_path2: The second RGB image file path.
    - save_path: The file path where the image should be saved.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band
        data = src.read(1)
    
    # Read the RGB images
    rgb_image1 = imread(rgb_path1)
    rgb_image2 = imread(rgb_path2)
    
    # Define a discrete color palette for the raster classes
    class_colors = ['#dae22f', '#6332ea', '#e346ee']  # Colors for Bare earth, NPV, PV, SI classes (low)
    # class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4']  # Colors for Bare earth, NPV, PV, SI classes (medium)
    # class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # Colors for Bare earth, NPV, PV, WI classes (dense)
    cmap = ListedColormap(class_colors)
    
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))  # Adjusted for three images
    
    # Display first RGB image
    axes[0].imshow(rgb_image1)
    # axes[0].set_title('RGB Imagery')
    axes[0].axis('off')  # Turn off axis numbers and ticks

    # Display second RGB image
    axes[1].imshow(rgb_image2)
    # axes[1].set_title('False Colour Composite - Multispectral Imagery')
    axes[1].axis('off')  # Turn off axis numbers and ticks

    # Display Binary Mask
    im = axes[2].imshow(data, interpolation='nearest', cmap=cmap)
    # axes[2].set_title('FVC Class Mask')
    axes[2].axis('off')  # Turn off axis numbers and ticks
    
    # Create a color bar to match the image height
    colorbar = fig.colorbar(im, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    ##low
    colorbar.set_ticks([0, 1, 2])  # Set ticks to match the number of classes #low
    colorbar.ax.set_yticklabels(['BE', 'NPV', 'PV'], fontsize=24)  # low
    ##medium
    # colorbar.set_ticks([0, 1, 2, 3])  # Set ticks to match the number of classes #medium
    # colorbar.ax.set_yticklabels(['BE class', 'NPV class', 'PV class', 'SI class'], fontsize=24)  # medium
    # ##dense
    # colorbar.set_ticks([0, 1, 2, 3])  # Set ticks to match the number of classes #dense
    # colorbar.ax.set_yticklabels(['BE class', 'NPV class', 'PV class', 'SI class', 'WI class'], fontsize=24)  #dense
    
    # Center the y-tick labels
    colorbar.ax.yaxis.set_ticks_position('right')
    colorbar.ax.yaxis.set_label_position('right')
    colorbar.ax.yaxis.set_ticks(colorbar.get_ticks())
    labels = colorbar.ax.get_yticklabels()
    colorbar.ax.set_yticklabels(labels, rotation=0, va='center')

    # Adjust layout to handle subplots nicely
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.1)  # Adjust the spacing between plots

    # Display plot
    # plt.show()

    # Uncomment below lines to save the figure to a file
    # plt.savefig(save_path)
    # plt.close(fig)  # Close the plot to free up memory
   
     # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    print(f"Image saved as {save_path}")

##low
rgb_file_path1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_rgb/tiles_rgb.33.tif'
rgb_file_path2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.33.tif'
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.33.tif'
output_image_path = '/home/laura/Documents/code/ecosystem_composition/phase_2_reference_data/3_pv_npv_be_classes/fvc_mask_low.png'


##medium
# rgb_file_path1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_rgb/tiles_rgb.22.tif'
# rgb_file_path2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.22.tif'
# raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.22.tif'
# output_image_path = '/home/laura/Documents/code/ecosystem_composition/phase_2_reference_data/3_pv_npv_be_classes/fvc_mask_medium.png'

##dense
# rgb_file_path1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/raw/tiles_rgb/tiles_rgb.118.tif'
# rgb_file_path2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.118.tif'
# raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.118.tif'
# output_image_path = '/home/laura/Documents/code/ecosystem_composition/phase_2_reference_data/3_pv_npv_be_classes/fvc_mask_dense.png'

plot_raster(raster_file_path, rgb_file_path1, rgb_file_path2, output_image_path)


#%%
# one image
import os
import warnings
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.errors import NotGeoreferencedWarning

def plot_mask_only(raster_path, save_path, mode='low'):
    """
    Plots a single classification mask raster with a color legend.
    - Values: 0 = BE, 1 = NPV, 2 = PV, [3 = SI (if mode=medium or dense)], NaN = white.
    - mode: 'low', 'medium', or 'dense' to control number of classes and colorbar.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(float)
            nodata_value = src.nodata
            if nodata_value is not None:
                data[data == nodata_value] = np.nan

    masked_data = np.ma.masked_invalid(data)

    # === Color setup by mode ===
    if mode == 'low':
        class_colors = ['#dae22f', '#6332ea', '#e346ee']  # BE, NPV, PV
        ticks = [0, 1, 2]
        labels = ['BE', 'NPV', 'PV']
        vmax = 2
    elif mode == 'medium':
        class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4']  # + SI
        ticks = [0, 1, 2, 3]
        labels = ['BE', 'NPV', 'PV', 'SI']
        vmax = 3
    elif mode == 'dense':
        class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # + SI, WI
        ticks = [0, 1, 2, 3, 4]
        labels = ['BE', 'NPV', 'PV', 'SI', 'WI']
        vmax = 5
    else:
        raise ValueError("mode must be 'low', 'medium', or 'dense'")

    cmap = ListedColormap(class_colors)
    cmap.set_bad('white')

    # === Plot ===
    fig, ax = plt.subplots(figsize=(20, 10))

    im = ax.imshow(masked_data, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels(labels, fontsize=24)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(f"Mask image saved as {save_path}")


#low
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.33.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/imgs/fvc_mask_low_3072_33.png'

#medium
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/imgs/fvc_mask_medium_3072_22.png'

#dense
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.118.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/imgs/fvc_mask_dense_3072_118.png'



plot_mask_only(raster_file_path, output_image_path, mode='low')
plot_mask_only(raster_file_path, output_image_path, mode='medium')
plot_mask_only(raster_file_path, output_image_path, mode='dense')


# %%
### be class (gradient)
import rasterio
import matplotlib.pyplot as plt

def plot_raster(raster_path, save_path):
    """
    Reads a raster file, displays it with a customized color bar, and saves the output image.
    Intended for visualization of binary class data such as 'Bare Earth' presence.

    Parameters:
    - raster_path: The file path to the raster file.
    - save_path: The file path where the image should be saved.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band
        data = src.read(1)
        
        # Get data range for color scaling
        vmin, vmax = data.min(), data.max()
        
        
        # Define a discrete color palette for the raster classes
        # class_colors = ['#ffffff', '#000000']  # Colors for Bare earth, NPV, PV, SI classes
        # cmap = ListedColormap(class_colors)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data, interpolation='nearest', cmap='grey')
        # ax.set_title('BE Class Mask')
        ax.axis('off')  # Turn off axis numbers and ticks
        
         # Create a color bar to match the image height
        colorbar = fig.colorbar(im, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
        colorbar.set_ticks([vmin, vmax])
        # colorbar.set_ticks([0, 1])  # Adjust ticks to match your data
        colorbar.ax.set_yticklabels(['Absence of BE Class', 'Presence of BE Class'], fontsize=14)  # Customize tick labels
        
        # Adjust layout to handle subplots nicely
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        # plt.show()
        # # Save the figure to a file
        plt.savefig(save_path)
        plt.close(fig)  # Close the plot to free up memory

        print(f"Image saved as {save_path}")


raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster/mask_be_globStat_percentile_tiles_multispectral.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/ecosystem_composition/phase_2_reference_data/step_2_fvc_classes_combined/be_mask.png'
plot_raster(raster_file_path, output_image_path)


#%%
### be class (msavi mask)
import rasterio
import matplotlib.pyplot as plt

def plot_raster(raster_path, save_path):
    """
    Reads a raster file, displays it with a customized color bar, and saves the output image.
    Intended for continuous data visualization such as MSAVI (Modified Soil Adjusted Vegetation Index).

    Parameters:
    - raster_path: The file path to the raster file.
    - save_path: The file path where the image should be saved.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band (assuming the raster is single-band)
        data = src.read(1)
        
        # Get the minimum and maximum values for color scaling
        vmin, vmax = data.min(), data.max()
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display the raster using a grey colormap
        im = ax.imshow(data, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')  # Turn off axis numbers and ticks
        
        # Create a color bar with continuous gradient matching the image's data range
        colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        
        # Set ticks to the min and max values and label them accordingly
        colorbar.set_ticks([vmin, vmax])
        colorbar.ax.set_yticklabels([f'Min: {vmin:.2f}', f'Max: {vmax:.2f}'], fontsize=16)
        
        # Adjust layout for better display
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        # Save the figure to a file
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the plot to free up memory

        print(f"Image saved as {save_path}")


# Example call to the function
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/msavi_globStat_percentile_tiles_multispectral.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/be_msavi.png'

plot_raster(raster_file_path, output_image_path)


#%%
### be class (binary mask)
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_raster(raster_path, save_path=None):
    """
    Plots a binary raster mask with a custom color palette and a corresponding color bar.
    Designed for visualizing binary classification from remote sensing data (e.g., Bare Earth class).

    Parameters:
    - raster_path: Path to the raster file (binary mask).
    - save_path: Path to save the output image. If None, the plot is displayed instead of saved.
    """
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band (binary mask)
        data = src.read(1)

    # Define a discrete color palette for the binary raster classes (0 and 1)
    class_colors = ['#ffffff', '#dae22f']  # Colors for BE class (0) and another class (1)
    cmap = ListedColormap(class_colors)

    # Create a plot to display the binary raster mask
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the binary raster mask
    im = ax.imshow(data, interpolation='nearest', cmap=cmap)
    ax.axis('off')  # Hide axes

    # Create a color bar with just two ticks for binary classes
    colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    colorbar.set_ticks([0, 0.5])  # Adjust ticks to match the binary class indices or chnge to 1 to put it at the top
    colorbar.ax.set_yticklabels(['Absence BE', 'Presence BE'], fontsize=16)  # Customize the labels

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided, otherwise show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        print(f"Image saved as {save_path}")
    else:
        plt.show()


# Example call to the function
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster/mask_be_globStat_percentile_tiles_multispectral.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/c_be_mask.png'

plot_raster(raster_file_path, output_image_path)



#%%
### be class (binary mask thresholded)
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_raster(raster_path, save_path=None):
    """
    Plots a binary raster mask where only the 'BE class' (value 0) is shown in a custom color,
    and all other values are ignored.

    Parameters:
    - raster_path: Path to the raster file (binary mask).
    - save_path: Path to save the output image. If None, the plot is displayed instead of saved.
    """
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band (binary mask)
        data = src.read(1)
        
        # Set all non-zero values to NaN to ignore them in the plot
        data = np.where(data == 0, 0, np.nan)

    # Define a color palette: Yellow for BE class (0), transparent for NaN (ignored areas)
    class_colors = ['#dae22f', '#ffffff']  # Yellow for BE class (0)
    cmap = ListedColormap([class_colors[0]])

    # Create a plot to display the raster mask (only BE class = 0)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the binary raster mask where only BE class (0) is visible
    im = ax.imshow(data, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.axis('off')  # Hide axes

    # Add a color bar (just for the BE class)
    colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    colorbar.set_ticks([0])  # Show only the BE class tick
    colorbar.ax.set_yticklabels(['BE Class'], fontsize=16)  # Customize the label

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided, otherwise show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        print(f"Image saved as {save_path}")
    else:
        plt.show()

# Example call to the function
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/be_mask_threshold.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/be_mask_thresholded.png'

plot_raster(raster_file_path, output_image_path)


#%%
### be class (binary mask thresholded improved)
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_raster(raster_path, save_path=None):
    """
    Plots a raster mask where '0' (presence of BE) is shown in yellow, and 'NaN' (absence) is shown in white.

    Parameters:
    - raster_path: Path to the raster file (binary mask).
    - save_path: Path to save the output image. If None, the plot is displayed instead of saved.
    """
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band (binary mask)
        data = src.read(1)
        
        # Retain 0 for presence and convert non-zero values to NaN
        data = np.where(data == 0, 0, np.nan)

    # Define a colormap: White for NaN, Yellow for 0
    class_colors = ['#dae22f']  # Yellow for presence
    cmap = ListedColormap(['#ffffff', '#dae22f'])  # White for absence, Yellow for presence

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the raster data with the colormap
    im = ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=-0.5, vmax=0.5)
    ax.axis('off')  # Hide axes

    # Add a color bar
    colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    # Adjust ticks to place "Presence (BE)" at the top and "Absence (NaN)" at the bottom
    colorbar.set_ticks([-0.5, 0])  # Ticks: [-0.5: NaN (Absence), 0: Presence]
    colorbar.ax.set_yticklabels(['Absence BE', 'Presence BE'], fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided, otherwise show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Image saved as {save_path}")
    else:
        plt.show()

# Example call to the function
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/be_mask_threshold.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/d_be_mask_thresholded.png'

plot_raster(raster_file_path, output_image_path)




#%% be class (composite image)
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.io import imread

def plot_raster(rgb_path, save_path=None):
    """
    Plots an RGB raster and displays a color bar representing the full color range of the image.

    Parameters:
    - rgb_path: Path to the RGB raster image.
    - save_path: Path to save the output image. If None, the plot is displayed instead of saved.
    """
    
    # Read the RGB image
    rgb_image = imread(rgb_path)

    # Create a plot to display the raster image
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the RGB image
    im = ax.imshow(rgb_image)
    ax.axis('off')  # Hide axes

    # Add a color bar for the image
    # colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    # colorbar.set_label('Color Intensity', fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided, otherwise show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        print(f"Image saved as {save_path}")
    else:
        plt.show()

# Example call to the function
# rgb_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.22.tif'
rgb_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_rgb/tiles_rgb.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/rgb_raster.png'

plot_raster(rgb_file_path, output_image_path)



# %%
#### improve msavi with rgb
import rasterio
import matplotlib.pyplot as plt
from skimage.io import imread

def plot_raster(raster_path, rgb_path, save_path):
    """
    Reads a raster file and an RGB image file, displays them side-by-side with a customized color bar for the raster,
    and saves the output image. The raster visualization is intended for binary class data such as 'Bare Earth' presence.

    Parameters:
    - raster_path: The file path to the raster file.
    - rgb_path: The file path to the RGB image file.
    - save_path: The file path where the image should be saved.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band
        data = src.read(1)
        
        # Ensure data is binary for correct visualization
        vmin, vmax = 0, 1  # Explicitly set for binary data
    
    # Read the RGB image
    rgb_image = imread(rgb_path)
    
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    # Display RGB image
    axes[0].imshow(rgb_image)
    # axes[0].set_title('False Colour Composite - Multispectral Imagery')
    axes[0].axis('off')  # Turn off axis numbers and ticks

    # Display Binary Mask
    im = axes[1].imshow(data, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
    # axes[1].set_title('BE Class Mask')
    axes[1].axis('off')  # Turn off axis numbers and ticks
    
    # Create a color bar to match the image height for the binary mask
    colorbar = fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    colorbar.set_ticks([vmin, vmax])
    colorbar.ax.set_yticklabels(['Absence of BE Class', 'Presence of BE Class'], fontsize=20)  # Customize tick labels
    
    # Adjust layout to handle subplots nicely
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.1)  # Adjust the spacing between plots

    # plt.show()
    # Save the figure to a file
    plt.savefig(save_path)
    plt.close(fig)  # Close the plot to free up memory

    print(f"Image saved as {save_path}")

# Example call to the function
raster_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster/mask_be_globStat_percentile_tiles_multispectral.22.tif'
# rgb_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.22.tif'
rgb_file_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_rgb/tiles_rgb.22.tif'
output_image_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_2_reference_data/step_2_fvc_classes_combined/be_mask.png'
plot_raster(raster_file_path, rgb_file_path, output_image_path)

# %%