## PHASE 2_Reference data
#STEP 1
# Step 1_3: BE class (from globStat_percentile) -MSAVI.TIF + MASK_BE.TIF + MASK_BE.SHP
######## get unique threshold for weight (VERSION 1)
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
import os


def msavi(nir_band, red_band):
    """
    Calculate the Modified Soil-Adjusted Vegetation Index (MSAVI).
    """
    return (2 * nir_band + 1 - np.sqrt((2 * nir_band + 1) ** 2 - 8 * (nir_band - red_band))) / 2

def calculate_msavi_for_multispectral_data(multispectral_image):
    """
    Calculates MSAVI for an entire multispectral image.
    """
    nir_band = multispectral_image[:, :, 9]  # Assuming Band 10 is NIR
    red_band = multispectral_image[:, :, 5]  # Assuming Band 6 is Red
    return msavi(nir_band, red_band)

def read_multispectral_image(filepath):
    """
    Reads a multispectral image file.
    """
    with rasterio.open(filepath) as src:
        image_data = src.read().transpose((1, 2, 0))
    return image_data

def save_image(data, reference_filepath, output_filepath):
    """
    Saves an image to a file, using a reference for the spatial profile.
    """
    with rasterio.open(reference_filepath) as src:
        profile = src.profile
        # Update the profile for single-band, ensuring dtype is float32
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        with rasterio.open(output_filepath, 'w', **profile) as dst:
            # Write the data as float32 directly, assuming binary mask values are 0 and 1
            dst.write(data.astype(rasterio.float32), 1)
            
def create_binary_mask(data):
    """
    Create a binary mask with values set to 1 for values at or above a weighted threshold
    designed to include low vegetation, and 0 for values below this threshold.

    Parameters:
    data (numpy array): The input data array.

    Returns:
    numpy array: A binary mask where values are 1 at or above the threshold and 0 below.
    """
    # Calculate the mean and median values of the data
    mean_value = np.mean(data)
    median_value = np.median(data)
    
    # Calculate a weighted threshold favoring the median to include more low vegetation [dense site12 + site08](exclude more of the lower vegetation areas)
    # weighted_threshold = (mean_value + 2 * median_value) / 3

    # Calculate a weighted threshold favoring the mean to include less low vegetation [medium site01 + site10_18] + [low site11]( more conservative and include only the higher vegetation densities)
    weighted_threshold = (2 * mean_value + median_value) / 3
    
    # Convert the weighted threshold to Decimal for precise rounding, then round to two decimal places
    rounded_threshold = Decimal(weighted_threshold.item()).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    # Convert the rounded threshold back to float for comparison with numpy array
    threshold = float(rounded_threshold)
    

    # Create the mask based on the rounded threshold
    mask_below_threshold = data <= threshold  # Values below the threshold are True (1)
    # mask_above_threshold = data > threshold  # Values at or above the threshold are False (0)
    
    return np.where(mask_below_threshold, 1, 0)  # Set 1 for True, 0 for False


def save_binary_mask_to_shapefile(mask, reference_filepath, output_shapefile_path, id_value=0, class_value='be'):
    """
    Save a binary raster mask as a shapefile with a single feature, representing the union of all mask areas.
    The feature has an 'id' of 0, a 'class' of 'be', and centroid coordinates 'x' and 'y'.
    """
    with rasterio.open(reference_filepath) as src:
        transform = src.transform
        crs = src.crs

    mask_shapes = shapes(mask.astype(np.int16), transform=transform)
    polygons = [shape(geom) for geom, value in mask_shapes if value == 1]
    
    if polygons:  # Check if there are any polygons
        # Merge all polygons into a single MultiPolygon
        merged_polygon = unary_union(polygons)

        if not merged_polygon.is_empty:  # Check if the merged polygon is not empty
            centroid = merged_polygon.centroid
            # Create a GeoDataFrame with a single feature
            gdf = gpd.GeoDataFrame({'id': [id_value], 'class': [class_value], 'x': [centroid.x], 'y': [centroid.y], 'geometry': [merged_polygon]}, crs=crs)
        else:
            # Handle the case of an empty merged polygon
            gdf = gpd.GeoDataFrame({'id': [id_value], 'class': [class_value], 'x': [None], 'y': [None], 'geometry': [None]}, crs=crs)
    else:
        # Handle the case with no polygons
        gdf = gpd.GeoDataFrame({'id': [id_value], 'class': [class_value], 'x': [None], 'y': [None], 'geometry': [None]}, crs=crs)
    
    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_shapefile_path)

              

# Define input and output folders
#site1_1 - supersite - DD0001 [MEDIUM]
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral/new'
output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/new'
output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster/new'
output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/new'

#site1_2 - similar to supersite - DD0010_18
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral'
# output_folder_msavi  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'

#site2_1 -  Vegetation - DD0011 [LOW] 
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/predictors/global_stats/tiles_multispectral'
# output_folder_msavi  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'

#site3_1 - with water - DD0012 [DENSE] 
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/predictors/glob_stats/tiles_multispectral'
# output_folder_msavi  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'

#site3_2 -similar to site 12 - DD0008 [DENSE]
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/predictors/glob_stats/tiles_multispectral'
# output_folder_msavi  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'


# Ensure the output folders exist
os.makedirs(output_folder_msavi, exist_ok=True)
os.makedirs(output_folder_binary, exist_ok=True)
os.makedirs(output_folder_shapefile, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):  # Check if the file is a TIF
        filepath = os.path.join(input_folder, filename)
        
        # Process the multispectral image
        multispectral_image = read_multispectral_image(filepath)
        msavi_values = calculate_msavi_for_multispectral_data(multispectral_image)
        
        # Calculate potential threshold values and log them
        mean_value = np.mean(msavi_values)
        median_value = np.median(msavi_values)
        percentile_99 = np.percentile(msavi_values, 99)
        
        print(f"File: {filepath}\nMean Value: {mean_value}\nMedian Value: {median_value}\n99th Percentile: {percentile_99}")
        
        # Construct the MSAVI output filepath with prefix "msavi_"
        # This assumes the input filename structure is "[filename].tif"
        msavi_output_filename = "msavi_" + filename
        msavi_output_filepath = os.path.join(output_folder_msavi, msavi_output_filename)
        
        # Save MSAVI output
        save_image(msavi_values, filepath, msavi_output_filepath)
        
        # Create and save the binary mask using the mean of MSAVI values as the threshold
        # binary_mask = create_binary_mask(msavi_values)
        #  pass the percentile_99_value to the function
        binary_mask = create_binary_mask(msavi_values)
        binary_mask_output_filename = "mask_be_" + filename  # Similarly, prefixing the binary mask file
        binary_mask_output_filepath = os.path.join(output_folder_binary, binary_mask_output_filename)
        save_image(binary_mask, filepath, binary_mask_output_filepath)
    
        print(f"Processed {filename}: MSAVI values saved as {msavi_output_filename} and binary mask saved.")
        
        # New: Save the binary mask as a shapefile
        shapefile_path = os.path.join(output_folder_shapefile, f"mask_be_{os.path.splitext(filename)[0]}.shp")
        save_binary_mask_to_shapefile(binary_mask, filepath, shapefile_path)

        print(f"MSAVI index image saved to {msavi_output_filepath}")
        print(f"Binary mask image saved to {binary_mask_output_filepath}")
        print(f"Shapefile saved to {shapefile_path}")

#%%
##############penalising weight (SELECTED VERSION 2)
######## get unique threshold for weight
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
import os


def msavi(nir_band, red_band):
    """
    Calculate the Modified Soil-Adjusted Vegetation Index (MSAVI).
    """
    return (2 * nir_band + 1 - np.sqrt((2 * nir_band + 1) ** 2 - 8 * (nir_band - red_band))) / 2

def calculate_msavi_for_multispectral_data(multispectral_image):
    """
    Calculates MSAVI for an entire multispectral image.
    """
    nir_band = multispectral_image[:, :, 9]  # Assuming Band 10 is NIR
    red_band = multispectral_image[:, :, 5]  # Assuming Band 6 is Red
    return msavi(nir_band, red_band)

def read_multispectral_image(filepath):
    """
    Reads a multispectral image file.
    """
    with rasterio.open(filepath) as src:
        image_data = src.read().transpose((1, 2, 0))
    return image_data

def save_image(data, reference_filepath, output_filepath):
    """
    Saves an image to a file, using a reference for the spatial profile.
    """
    with rasterio.open(reference_filepath) as src:
        profile = src.profile
        # Update the profile for single-band, ensuring dtype is float32
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        with rasterio.open(output_filepath, 'w', **profile) as dst:
            # Write the data as float32 directly, assuming binary mask values are 0 and 1
            dst.write(data.astype(rasterio.float32), 1)

#%%           
def create_binary_mask(data):
    """
    Create a binary mask with values set to 1 for values at or above a weighted threshold
    designed to include low vegetation, and 0 for values below this threshold.

    Parameters:
    data (numpy array): The input data array.

    Returns:
    numpy array: A binary mask where values are 1 at or above the threshold and 0 below.
    """
    # Calculate the mean and median values of the data
    mean_value = np.mean(data)
    median_value = np.median(data)
    
    # Calculate a weighted threshold favoring the median to include more low vegetation [dense site12 + site08](exclude more of the lower vegetation areas)
    # weighted_threshold = (mean_value + 2 * median_value) / 3 ##good

    # Calculate a weighted threshold favoring the mean to include less low vegetation [medium site01 + site10_18] ( more conservative and include only the higher vegetation densities)
    # weighted_threshold = (2 * mean_value + median_value) / 3 ##old  
    
    # Adjust the weighted threshold to be slightly below the median ##correct one
    weighted_threshold = (3 * median_value + mean_value) / 4 
    
    # Ensure that the calculated threshold doesn't accidentally exceed the median
    if weighted_threshold > median_value:
        weighted_threshold = median_value - 0.03  # Slightly less than the median --> 0.03 (dense) / 0.03 (medium) / 0.06 (low)
    
    # Convert the weighted threshold to Decimal for precise rounding, then round to two decimal places
    rounded_threshold = Decimal(weighted_threshold.item()).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    # Convert the rounded threshold back to float for comparison with numpy array
    threshold = float(rounded_threshold)
    

    # Create the mask based on the rounded threshold
    mask_above_threshold = data >= threshold  # Values at or above the threshold are True (1)
    # mask_above_threshold = data > threshold  # Values at or above the threshold are False (0)
    
    return np.where(mask_above_threshold, 0, 1)  # Set 1 for True, 0 for False


def save_binary_mask_to_shapefile(mask, reference_filepath, output_shapefile_path, id_value=0, class_value='be'):
    """
    Save a binary raster mask as a shapefile with a single feature, representing the union of all mask areas.
    The feature has an 'id' of 0, a 'class' of 'be', and centroid coordinates 'x' and 'y'.
    """
    with rasterio.open(reference_filepath) as src:
        transform = src.transform
        crs = src.crs

    mask_shapes = shapes(mask.astype(np.int16), transform=transform)
    polygons = [shape(geom) for geom, value in mask_shapes if value == 1]
    
    if polygons:  # Check if there are any polygons
        # Merge all polygons into a single MultiPolygon
        merged_polygon = unary_union(polygons)

        if not merged_polygon.is_empty:  # Check if the merged polygon is not empty
            centroid = merged_polygon.centroid
            # Create a GeoDataFrame with a single feature
            gdf = gpd.GeoDataFrame({'id': [id_value], 'class': [class_value], 'x': [centroid.x], 'y': [centroid.y], 'geometry': [merged_polygon]}, crs=crs)
        else:
            # Handle the case of an empty merged polygon
            gdf = gpd.GeoDataFrame({'id': [id_value], 'class': [class_value], 'x': [None], 'y': [None], 'geometry': [None]}, crs=crs)
    else:
        # Handle the case with no polygons
        gdf = gpd.GeoDataFrame({'id': [id_value], 'class': [class_value], 'x': [None], 'y': [None], 'geometry': [None]}, crs=crs)
    
    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_shapefile_path)

              

# Define input and output folders
#site1_1 - supersite - DD0001 [MEDIUM]
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral'
output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/new'

#site1_2 - similar to supersite - DD0010_18
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral'
# output_folder_msavi  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'

#site2_1 -  Vegetation - DD0011 [LOW] 
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/global_stats/tiles_multispectral'
# output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'

#site3_1 - with water - DD0012 [DENSE] 
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral'
# output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'

#site3_2 -similar to site 12 - DD0008 [DENSE]
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/predictors/glob_stats/tiles_multispectral'
# output_folder_msavi  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'


# Ensure the output folders exist
os.makedirs(output_folder_msavi, exist_ok=True)
os.makedirs(output_folder_binary, exist_ok=True)
os.makedirs(output_folder_shapefile, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):  # Check if the file is a TIF
        filepath = os.path.join(input_folder, filename)
        
        # Process the multispectral image
        multispectral_image = read_multispectral_image(filepath)
        msavi_values = calculate_msavi_for_multispectral_data(multispectral_image)
        
        # Calculate potential threshold values and log them
        mean_value = np.mean(msavi_values)
        median_value = np.median(msavi_values)
        percentile_99 = np.percentile(msavi_values, 99)
        
        print(f"File: {filepath}\nMean Value: {mean_value}\nMedian Value: {median_value}\n99th Percentile: {percentile_99}")
        
        # Construct the MSAVI output filepath with prefix "msavi_"
        # This assumes the input filename structure is "[filename].tif"
        msavi_output_filename = "msavi_" + filename
        msavi_output_filepath = os.path.join(output_folder_msavi, msavi_output_filename)
        
        # Save MSAVI output
        save_image(msavi_values, filepath, msavi_output_filepath)
        
        # Create and save the binary mask using the mean of MSAVI values as the threshold
        # binary_mask = create_binary_mask(msavi_values)
        #  pass the percentile_99_value to the function
        binary_mask = create_binary_mask(msavi_values)
        binary_mask_output_filename = "mask_be_" + filename  # Similarly, prefixing the binary mask file
        binary_mask_output_filepath = os.path.join(output_folder_binary, binary_mask_output_filename)
        save_image(binary_mask, filepath, binary_mask_output_filepath)
    
        print(f"Processed {filename}: MSAVI values saved as {msavi_output_filename} and binary mask saved.")
        
        # New: Save the binary mask as a shapefile
        shapefile_path = os.path.join(output_folder_shapefile, f"mask_be_{os.path.splitext(filename)[0]}.shp")
        save_binary_mask_to_shapefile(binary_mask, filepath, shapefile_path)

        print(f"MSAVI index image saved to {msavi_output_filepath}")
        print(f"Binary mask image saved to {binary_mask_output_filepath}")
        print(f"Shapefile saved to {shapefile_path}")

#%%
##################
### print threshold
import os
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import rasterio

# Define file paths
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi'  # Replace with your folder path
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi'

def print_thresholds_for_msavi_files(folder_path):
    """
    Processes each MSAVI file in a folder, calculates and prints the weighted threshold for each file.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            
            # Open the MSAVI file and read the data
            with rasterio.open(file_path) as src:
                msavi_data = src.read(1)  # Assume MSAVI data is in the first band
            
            # Calculate the mean and median
            mean_value = np.mean(msavi_data)
            median_value = np.median(msavi_data)
            
            # Calculate the weighted threshold
            # weighted_threshold = (2 * mean_value + median_value) / 3
            weighted_threshold = (3 * median_value + mean_value) / 4 
            threshold = float(Decimal(weighted_threshold).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            
            print(f"Threshold for {filename}: {threshold}")

# Run the function on the MSAVI files folder
print_thresholds_for_msavi_files(input_folder)

# %%
import os
import numpy as np
import rasterio

# Define the input folder containing the MSAVI files
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi'  # Replace with your folder path
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi'


def calculate_overall_statistics(folder_path):
    """
    Calculates the overall mean, median, min, max, and interquartile range (IQR) for MSAVI data across all files in a folder.
    Also calculates and prints min, max, and weighted threshold values for each file.
    """
    all_data = []  # List to accumulate all pixel values from each file
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            
            # Open the MSAVI file and read the data
            with rasterio.open(file_path) as src:
                msavi_data = src.read(1).flatten()  # Flatten to 1D array for easier concatenation
                all_data.extend(msavi_data)  # Append to the overall data list
                
                # Calculate min, max, and threshold for the current file
                file_min = np.min(msavi_data)
                file_max = np.max(msavi_data)
                mean_value = np.mean(msavi_data)
                median_value = np.median(msavi_data)
                
                # Calculate the weighted threshold
                weighted_threshold = (2 * mean_value + median_value) / 3
                threshold = float(Decimal(weighted_threshold).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                
                # Print individual file statistics
                print(f"File: {filename} - Min: {file_min}, Max: {file_max}, Threshold: {threshold}")

    # Convert all_data to a numpy array for overall calculations
    all_data = np.array(all_data)
    
    # Calculate overall statistics
    overall_mean = np.mean(all_data)
    overall_median = np.median(all_data)
    overall_min = np.min(all_data)
    overall_max = np.max(all_data)
    q1 = np.percentile(all_data, 25)
    q3 = np.percentile(all_data, 75)
    iqr = q3 - q1  # Interquartile range
    
    # Print the overall statistics
    print("\nOverall Statistics:")
    print(f"Overall Mean: {overall_mean}")
    print(f"Overall Median: {overall_median}")
    print(f"Overall Min: {overall_min}")
    print(f"Overall Max: {overall_max}")
    print(f"Interquartile Range (IQR): {iqr}")

# Run the function on the MSAVI files folder
calculate_overall_statistics(input_folder)


# %%
import os
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import rasterio

# Define the input folder containing the MSAVI files
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi'  # Replace with your folder path
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi'

# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'  # Replace with your folder path
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi'



def calculate_overall_statistics(folder_path):
    """
    Calculates the overall mean, median, min, max, and interquartile range (IQR) for MSAVI data across all files in a folder.
    Also calculates and prints min, max, mean, median, and IQR of the weighted threshold values across all files.
    """
    all_data = []  # List to accumulate all pixel values from each file
    thresholds = []  # List to accumulate threshold values from each file
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            
            # Open the MSAVI file and read the data
            with rasterio.open(file_path) as src:
                msavi_data = src.read(1).flatten()  # Flatten to 1D array for easier concatenation
                all_data.extend(msavi_data)  # Append to the overall data list
                
                # Calculate min, max, and threshold for the current file
                file_min = np.min(msavi_data)
                file_max = np.max(msavi_data)
                mean_value = np.mean(msavi_data)
                median_value = np.median(msavi_data)
                
                # Calculate the weighted threshold
                # weighted_threshold = (2 * mean_value + median_value) / 3
                weighted_threshold = (3 * median_value + mean_value) / 4 
                if weighted_threshold > median_value:
                    weighted_threshold = median_value - 0.03  # Slightly less than the median --> 0.03 (dense) / 0.03 (medium) / 0.06 (low)
                threshold = float(Decimal(weighted_threshold).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                thresholds.append(threshold)  # Store the threshold for min/max/mean/median/IQR calculation
                
                # Print individual file statistics
                print(f"File: {filename} - Min: {file_min}, Max: {file_max}, Threshold: {threshold}")

    # Convert all_data to a numpy array for overall calculations
    all_data = np.array(all_data)
    
    # Calculate overall statistics for MSAVI data
    overall_mean = np.mean(all_data)
    overall_median = np.median(all_data)
    overall_min = np.min(all_data)
    overall_max = np.max(all_data)
    q1, q2, q3 = np.percentile(all_data, [25, 50, 75])
    iqr = q3 - q1  # Interquartile range for MSAVI data

    
    # Calculate the min, max, mean, median, and IQR of all thresholds
    threshold_min = min(thresholds)
    threshold_max = max(thresholds)
    threshold_mean = np.mean(thresholds)
    threshold_median = np.median(thresholds)
    threshold_q1, threshold_q2, threshold_q3 = np.percentile(thresholds, [25, 50, 75])
    threshold_iqr = threshold_q3 - threshold_q1  # Interquartile range for thresholds
    
    # Print the overall statistics
    print("\nOverall Statistics for MSAVI Data:")
    print(f"Overall Mean: {overall_mean}")
    print(f"Overall Median: {overall_median}")
    print(f"Overall Min: {overall_min}")
    print(f"Overall Max: {overall_max}")
    print(f"Interquartile Range (IQR): {iqr}")
    
    print("\nThreshold Statistics Across Files:")
    print(f"Min Threshold: {threshold_min}")
    print(f"Max Threshold: {threshold_max}")
    print(f"Mean Threshold: {threshold_mean}")
    print(f"Median Threshold: {threshold_median}")
    print(f"Interquartile Range (IQR) of Thresholds: {threshold_iqr}")

# Run the function on the MSAVI files folder
calculate_overall_statistics(input_folder)

# %%
