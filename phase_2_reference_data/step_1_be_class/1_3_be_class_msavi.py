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
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral/clean'
output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'
output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_shp'

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
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral/clean'
# output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
# output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'
# output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_shp'


#site1_2 - similar to supersite - DD0010_18
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral'
# output_folder_msavi  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/msavi'
# output_folder_binary  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/annotation_raster'
# output_folder_shapefile = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp'

#site2_1 -  Vegetation - DD0011 [LOW] 
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/global_stats/tiles_multispectral/clean'
# output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
# output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'
# output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_shp'

#site3_1 - with water - DD0012 [DENSE] 
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral/clean'
output_folder_msavi  = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
output_folder_binary  = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'
output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_shp'

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
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'

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


#%%
### Calculate NDVI spectral index
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
import os

def ndvi(nir_band, red_band):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).
    """
    return (nir_band - red_band) / (nir_band + red_band)

def calculate_ndvi_for_multispectral_data(multispectral_image):
    """
    Calculates NDVI for an entire multispectral image.
    """
    nir_band = multispectral_image[:, :, 9]  # Assuming Band 10 is NIR
    red_band = multispectral_image[:, :, 5]  # Assuming Band 6 is Red
    return ndvi(nir_band, red_band)

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
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        with rasterio.open(output_filepath, 'w', **profile) as dst:
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


# === Set up paths ===
# Medium
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral/clean'
# output_folder_ndvi = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi'
# output_folder_binary = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
# output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_shp'

# Low
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/global_stats/tiles_multispectral/clean'
# output_folder_ndvi = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi'
# output_folder_binary = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
# output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_shp'

# dense
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/glob_stats/tiles_multispectral/clean'
output_folder_ndvi = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/ndvi'
output_folder_binary = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
output_folder_shapefile = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_shp'


os.makedirs(output_folder_ndvi, exist_ok=True)
os.makedirs(output_folder_binary, exist_ok=True)
os.makedirs(output_folder_shapefile, exist_ok=True)


# === Process all tiles ===
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        filepath = os.path.join(input_folder, filename)
        print(f"\nProcessing {filename}...")

        try:
            multispectral_image = read_multispectral_image(filepath)
            ndvi_image = calculate_ndvi_for_multispectral_data(multispectral_image)

            # Stats for logging
            mean_value = np.mean(ndvi_image)
            median_value = np.median(ndvi_image)
            percentile_99 = np.percentile(ndvi_image, 99)

            print(f"Mean NDVI: {mean_value:.3f}, Median: {median_value:.3f}, 99th Percentile: {percentile_99:.3f}")

            # Save NDVI image
            ndvi_output_filename = "ndvi_" + filename
            ndvi_output_path = os.path.join(output_folder_ndvi, ndvi_output_filename)
            save_image(ndvi_image, filepath, ndvi_output_path)

            # Create and save binary mask
            binary_mask = create_binary_mask(ndvi_image)
            binary_output_filename = "mask_be_" + filename
            binary_output_path = os.path.join(output_folder_binary, binary_output_filename)
            save_image(binary_mask, filepath, binary_output_path)

            # Save shapefile
            shapefile_path = os.path.join(output_folder_shapefile, f"mask_be_{os.path.splitext(filename)[0]}.shp")
            save_binary_mask_to_shapefile(binary_mask, filepath, shapefile_path)

            print(f"Saved NDVI: {ndvi_output_path}")
            print(f"Saved Binary Mask: {binary_output_path}")
            print(f"Saved Shapefile: {shapefile_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")




# %%
# Calculate stats
import os
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import rasterio

# Define the input folder containing the 
# MSAVI files
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean'
# input_folder= '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean'  # Replace with your folder path
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean'

# NDVI file
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi'

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


#%%
### Calculate stats improve previous when with nan values appeared
import os
import numpy as np
import rasterio
from decimal import Decimal, ROUND_HALF_UP

# MSAVI files
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean'
# input_folder= '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean/check'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean'  # Replace with your folder path
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/clean'

# NDVI file
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi'
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi'


def calculate_overall_statistics(folder_path):
    """
    Calculates robust statistics (mean, median, min, max, IQR) for MSAVI values and thresholds across all .tif files.
    Ignores invalid tiles (all NaNs) and applies a logic rule to adjust the weighted threshold slightly below the median.
    """
    all_data = []
    thresholds = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)

            with rasterio.open(file_path) as src:
                msavi_data = src.read(1).astype(np.float32).flatten()
                valid_data = msavi_data[~np.isnan(msavi_data)]

                if valid_data.size == 0:
                    print(f"File: {filename} - Skipped (all values are NaN)")
                    continue

                file_min = np.nanmin(valid_data)
                file_max = np.nanmax(valid_data)
                mean_value = np.nanmean(valid_data)
                median_value = np.nanmedian(valid_data)

                # Custom weighted threshold formula
                weighted_threshold = (3 * median_value + mean_value) / 4
                if weighted_threshold > median_value:
                    weighted_threshold = median_value - 0.03  # Adjust based on vegetation density logic

                threshold = float(Decimal(weighted_threshold).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                thresholds.append(threshold)
                all_data.extend(valid_data)

                print(f"File: {filename} - Min: {file_min:.4f}, Max: {file_max:.4f}, Threshold: {threshold:.2f}")

    # Convert to numpy arrays for final calculations
    all_data = np.array(all_data)
    thresholds = np.array(thresholds)

    if all_data.size == 0:
        print("\nNo valid MSAVI data found in folder.")
        return

    # Overall MSAVI stats
    overall_mean = np.nanmean(all_data)
    overall_median = np.nanmedian(all_data)
    overall_min = np.nanmin(all_data)
    overall_max = np.nanmax(all_data)
    q1, q3 = np.nanpercentile(all_data, [25, 75])
    iqr = q3 - q1

    # Threshold stats
    threshold_min = np.nanmin(thresholds)
    threshold_max = np.nanmax(thresholds)
    threshold_mean = np.nanmean(thresholds)
    threshold_median = np.nanmedian(thresholds)
    tq1, tq3 = np.nanpercentile(thresholds, [25, 75])
    threshold_iqr = tq3 - tq1

    # Print results
    print("\nOverall Statistics for MSAVI Data:")
    print(f"Mean: {overall_mean:.4f}")
    print(f"Median: {overall_median:.4f}")
    print(f"Min: {overall_min:.4f}")
    print(f"Max: {overall_max:.4f}")
    print(f"Interquartile Range (IQR): {iqr:.4f}")

    print("\nThreshold Statistics Across Files:")
    print(f"Min Threshold: {threshold_min:.2f}")
    print(f"Max Threshold: {threshold_max:.2f}")
    print(f"Mean Threshold: {threshold_mean:.2f}")
    print(f"Median Threshold: {threshold_median:.2f}")
    print(f"Interquartile Range (IQR): {threshold_iqr:.2f}")


# Run the function on the MSAVI files folder
calculate_overall_statistics(input_folder)


# %%
#### Compare MSAVI vs NDVI with Intersection of Union (IoU)
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Paths to shapefiles
#medium
# ndvi_shp_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_shp'
# msavi_shp_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_shp'

#low
# ndvi_shp_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_shp'
# msavi_shp_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_shp'

#dense
ndvi_shp_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_shp'
msavi_shp_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_shp'


results = []

for filename in os.listdir(ndvi_shp_dir):
    if filename.endswith(".shp"):
        ndvi_path = os.path.join(ndvi_shp_dir, filename)
        msavi_path = os.path.join(msavi_shp_dir, filename)

        if not os.path.exists(msavi_path):
            print(f"Missing MSAVI shapefile for: {filename}")
            continue

        try:
            gdf_ndvi = gpd.read_file(ndvi_path)
            gdf_msavi = gpd.read_file(msavi_path)

            # Drop empty or null geometries
            gdf_ndvi = gdf_ndvi[~gdf_ndvi.geometry.is_empty & gdf_ndvi.geometry.notnull()]
            gdf_msavi = gdf_msavi[~gdf_msavi.geometry.is_empty & gdf_msavi.geometry.notnull()]

            if gdf_ndvi.empty or gdf_msavi.empty:
                iou = 0.0
            else:
                union_ndvi = gdf_ndvi.geometry.union_all()
                union_msavi = gdf_msavi.geometry.union_all()

                intersection_area = union_ndvi.intersection(union_msavi).area
                union_area = union_ndvi.union(union_msavi).area

                iou = intersection_area / union_area if union_area > 0 else 0.0

            results.append({
                'tile': filename,
                'Polygon_IoU': round(iou, 4)
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert to DataFrame
df_iou = pd.DataFrame(results)
df_iou.sort_values('Polygon_IoU', ascending=False, inplace=True)

# Save results
df_iou.to_csv('polygon_iou_ndvi_vs_msavi.csv', index=False)

# === Print Summary Stats ===
print("\n=== Polygon IoU Summary ===")
print(f"Total tiles compared: {len(df_iou)}")
print(f"Mean IoU:   {df_iou['Polygon_IoU'].mean():.4f}")
print(f"Median IoU: {df_iou['Polygon_IoU'].median():.4f}")
print(f"Min IoU:    {df_iou['Polygon_IoU'].min():.4f}")
print(f"Max IoU:    {df_iou['Polygon_IoU'].max():.4f}")

# Preview results
print("\nTop 5 highest IoU tiles:")
print(df_iou.head())


#%%
#### Compare MSAVI vs NDVI at pixel level (option 1)
import os
import numpy as np
import rasterio
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score, accuracy_score

# Paths to masks
#medium
# ndvi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
# msavi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'

#low
# ndvi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
# msavi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'

#dense
ndvi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
msavi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'


# Initialize results
results = []

# Loop through NDVI mask files
for filename in os.listdir(ndvi_mask_dir):
    if filename.endswith('.tif'):
        ndvi_path = os.path.join(ndvi_mask_dir, filename)
        msavi_path = os.path.join(msavi_mask_dir, filename)

        if not os.path.exists(msavi_path):
            print(f"Missing MSAVI mask for: {filename}")
            continue

        try:
            with rasterio.open(ndvi_path) as src1, rasterio.open(msavi_path) as src2:
                ndvi_mask = src1.read(1)
                msavi_mask = src2.read(1)

            # Flatten and mask invalid values if needed
            ndvi_mask_flat = ndvi_mask.flatten()
            msavi_mask_flat = msavi_mask.flatten()

            # Ensure same shape
            if ndvi_mask_flat.shape != msavi_mask_flat.shape:
                print(f"Shape mismatch in {filename}")
                continue

            # Compute metrics
            iou = jaccard_score(ndvi_mask_flat, msavi_mask_flat, pos_label=1)
            acc = accuracy_score(ndvi_mask_flat, msavi_mask_flat)
            prec = precision_score(ndvi_mask_flat, msavi_mask_flat, pos_label=1)
            rec = recall_score(ndvi_mask_flat, msavi_mask_flat, pos_label=1)
            f1 = f1_score(ndvi_mask_flat, msavi_mask_flat, pos_label=1)

            results.append({
                'tile': filename,
                'IoU': iou,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert to DataFrame and save
df_results = pd.DataFrame(results)
df_results.sort_values('IoU', ascending=False, inplace=True)

# Print summary
print(df_results.head())

# Save to CSV
df_results.to_csv('ndvi_vs_msavi_mask_comparison.csv', index=False)


# %%
#### Compare MSAVI vs NDVI at pixel level (option 2)
import os
import rasterio
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Define input directories
#medium
# ndvi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
# msavi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'

#low
# ndvi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
# msavi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'

#dense
ndvi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/ndvi/annotation_raster'
msavi_mask_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/msavi/annotation_raster'

pixel_size_m = 0.01  # change if your pixel size is not 5cm

results = []

for filename in os.listdir(ndvi_mask_dir):
    if filename.endswith('.tif'):
        ndvi_path = os.path.join(ndvi_mask_dir, filename)
        msavi_path = os.path.join(msavi_mask_dir, filename)

        if not os.path.exists(msavi_path):
            print(f"Missing MSAVI mask for: {filename}")
            continue

        try:
            with rasterio.open(ndvi_path) as ndvi_src, rasterio.open(msavi_path) as msavi_src:
                ndvi_mask = ndvi_src.read(1)
                msavi_mask = msavi_src.read(1)

            # Flatten and filter out nodata
            valid_mask = (ndvi_mask >= 0) & (msavi_mask >= 0)
            ndvi_flat = ndvi_mask[valid_mask].flatten()
            msavi_flat = msavi_mask[valid_mask].flatten()

            if len(ndvi_flat) == 0 or len(msavi_flat) == 0:
                continue

            # Confusion matrix & metrics
            cm = confusion_matrix(ndvi_flat, msavi_flat, labels=[1, 0])
            acc = accuracy_score(ndvi_flat, msavi_flat)
            prec = precision_score(ndvi_flat, msavi_flat, pos_label=1)
            rec = recall_score(ndvi_flat, msavi_flat, pos_label=1)
            f1 = f1_score(ndvi_flat, msavi_flat, pos_label=1)
            kappa = cohen_kappa_score(ndvi_flat, msavi_flat)

            # Disagreement mask
            disagreement = np.sum(ndvi_flat != msavi_flat)
            total_pixels = len(ndvi_flat)
            pct_disagree = (disagreement / total_pixels) * 100

            # Area difference (bare ground = 1)
            ndvi_area = np.sum(ndvi_flat == 1)
            msavi_area = np.sum(msavi_flat == 1)
            pixel_area_m2 = pixel_size_m ** 2
            area_diff_m2 = (msavi_area - ndvi_area) * pixel_area_m2

            results.append({
                'tile': filename,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1 Score': round(f1, 4),
                'Kappa': round(kappa, 4),
                '% Disagree': round(pct_disagree, 2),
                'NDVI_BG_Pixels': ndvi_area,
                'MSAVI_BG_Pixels': msavi_area,
                'Area_Diff_m2': round(area_diff_m2, 2)
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert to DataFrame
df = pd.DataFrame(results)
df.sort_values('F1 Score', ascending=False, inplace=True)

# Print summary
print("\n--- NDVI vs MSAVI Per-Pixel Agreement Summary ---")
print(df.describe())

# Save to CSV
df.to_csv('pixel_metrics_ndvi_vs_msavi.csv', index=False)

# %%
