#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:46:42 2023

@author: lauransotomayor
"""

#%%
## PHASE 1_Image Processing
#STEP 1 (option a)
# Step 1_1: Reproject/Upsample from 5cm to 1cm by using bilinear resampling method
# Check resolution

from osgeo import gdal
import rasterio
import subprocess

# Define the path to your input raster file
#site1_1 - supersite [MEDIUM]  DONE ~26min
# input_image_fn = '/media/laura/Extreme SSD/uas_data/Calperum/SASMDD0001/20220519/micasense_dual/level_1/20220519_SASMDD0001_dual_ortho_05.tif'
# output_raster_fn = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_supersite_DD0001/20220519_SASMDD001_dual_ortho_01_bilinear.tif'

#site2_1 -  Vegetation [LOW] (potentially site DD0013)
input_image_fn = '/media/laura/Extreme SSD/uas_data/Calperum/SASMDD0011/20220517/micasense_dual/level_1/20220517_SASMDD0011_dual_ortho_06.tif'
output_raster_fn = '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_01_bilinear.tif'

#site3_1 - with water  [DENSE] DONE
# input_image_fn = '/media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif'
# output_raster_fn = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/20220517_SASMDD0012_dual_ortho_01_bilinear.tif'


# Extract the output folder path from the output raster file name
output_folder = os.path.dirname(output_raster_fn)

# Create the output directory if it doesn't already exist
os.makedirs(output_folder, exist_ok=True)

# get the bounds of the input image -- we use these to 
with rasterio.open(input_image_fn) as f:
    left, bottom, right, top = f.bounds
    width = f.width
    height = f.height
    crs = f.crs
    # print(f.crs)

    
# Define the gdalwarp command as a list of arguments
gdal_rasterize_command = [
    'gdalwarp',
    '-overwrite',  # Ensures existing files are overwritten
    '-s_srs', str(crs),
    '-t_srs', 'EPSG:7854', # Target CRS
    '-tr', '0.01', '0.01', # Target resolution in meters (1cm)
    '-r', 'bilinear',
    '-te', str(left), str(bottom), str(right), str(top),  # the output GeoTIFF should cover the same bounds as the input image
    '-te_srs', str(crs),
    # '-ts', str(width), str(height),  # the output GeoTIFF should have the same height and width as the input image
    '-ot', 'Float32',
    '-of', 'GTiff',
    '-co', 'COMPRESS=LZW',  # compress it
    '-co', 'PREDICTOR=2',  # compress it good
    '-co', 'BIGTIFF=YES',  # just incase the image is bigger than 4GB
    '-dstnodata', '-32767',  # Set NoData value
    input_image_fn,
    output_raster_fn
]


# Execute the gdal_rasterize command using subprocess
try:
    subprocess.run(gdal_rasterize_command, check=True)
    print("Rasterization completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error executing gdal_rasterize: {e}")
 

 # Creating output file that is 35620P x 38825L.
 # Processing /media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif [1/1] : 0Using internal nodata values (e.g. -32767) for image /media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif.
 # Copying nodata values from source /media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif to destination /media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/20220517_SASMDD0012_dual_ortho_01_bilinear.tif.
 # ...10...20...30...40...50...60...70...80...90...100 - done.
 # Rasterization completed successfully.


#%%
###################### Dask Distrution (slighlty faster)
#STEP 1 (option b)
# Step 1_1: Reproject/Downscale from 5cm to 1cm by using bilinear resampling method
'''Apply parallel processing for reprojection and resampling of raster images using Dask.'''
import argparse
import subprocess
from dask.distributed import Client, as_completed
import rasterio
import time  # Import the time module
import os

# Define the path to your input raster file
#site1 - supersite [MEDIUM] DONE
# input_image_fn = '/media/laura/Extreme SSD/uas_data/Calperum/SASMDD0001/20220519/micasense_dual/level_1/20220519_SASMDD0001_dual_ortho_05.tif'
# output_raster_fn = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_supersite_DD0001/20220519_SASMDD001_dual_ortho_01_bilinear.tif'

#site2 - similar to supersite [MEDIUM] DONE ~22min
# input_image_fn = '/media/laura/Extreme SSD/uas_data/Calperum/SASMD0010_18/20220518/micasense_dual/level_1/20220518_SASMDD0010_18_dual_ortho_05.tif'
# output_raster_fn = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/20220518_SASMDD0010_18_dual_ortho_01_bilinear.tif'

# site3 - with water  [DENSE] DONE
# input_image_fn = '/media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif'
# output_raster_fn = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/20220517_SASMDD0012_dual_ortho_01_bilinear.tif'

# site4 - Vegetation [LOW] (potentially site DD0013)
input_image_fn = '/media/laura/Extreme SSD/uas_data/Calperum/SASMDD0011/20220517/micasense_dual/level_1/20220517_SASMDD0011_dual_ortho_06.tif'
output_raster_fn = '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_01_bilinear.tif'


def reproject_and_resample(input_image_fn, output_raster_fn, target_crs, resolution, resampling_method):
    """Function to reproject and resample a single image."""
    # Check if output file exists and delete it to avoid gdalwarp error
    if os.path.exists(output_raster_fn):
        os.remove(output_raster_fn)
    
    with rasterio.open(input_image_fn) as f:
        bounds = f.bounds
        crs = f.crs

    gdal_rasterize_command = [
        'gdalwarp',
        '-overwrite',  # Ensures existing files are overwritten
        '-s_srs', str(crs),
        '-t_srs', target_crs,
        '-tr', str(resolution), str(resolution),
        '-r', resampling_method,
        '-te', str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
        '-ot', 'Float32',
        '-of', 'GTiff',
        '-co', 'COMPRESS=LZW',
        '-co', 'PREDICTOR=2',
        '-co', 'BIGTIFF=YES',
        '-dstnodata', '-32767',  # Set NoData value
        input_image_fn,
        output_raster_fn
    ]

    # Execute the gdalwarp command
    try:
        subprocess.run(gdal_rasterize_command, check=True)
        return ["Rasterization completed successfully."]
    except subprocess.CalledProcessError as e:
        return [f"Error executing gdalwarp: {e}"]
    

def process_images_in_parallel(image_pairs, target_crs, resolution, resampling_method):
    client = Client()  # Starts a Dask client; it automatically manages port issues

    start_time = time.time()

    futures = [client.submit(reproject_and_resample, img[0], img[1], target_crs, resolution, resampling_method) for img in image_pairs]

    results = []
    for future in as_completed(futures):
        result = future.result()  # Ensure the result is iterable
        results.extend(result)  # Extend the results list with the result of the future

    client.close()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    return results

    
if __name__ == '__main__':
    
    # Example usage - this part should be adapted based on your actual workflow
    image_pairs = [
        (input_image_fn, output_raster_fn),
        # Uncomment the following line to include site3 in the processing
        # (input_image_fn_site3, output_raster_fn_site3),
    ]
    target_crs = 'EPSG:7854'
    resolution = 0.01
    resampling_method = 'bilinear'
    
    # Process images in parallel
    results = process_images_in_parallel(image_pairs, target_crs, resolution, resampling_method)
    for result in results:
        print(result)  # Print the result of each operation


# Creating output file that is 35620P x 38825L.
# Processing /media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif [1/1] : 0Using internal nodata values (e.g. -32767) for image /media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif.
# Copying nodata values from source /media/laura/Extreme SSD/uas_data/Calperum/SASMDD0012/20220517/micasense_dual/level_1/20220517_SASMDD0012_dual_ortho_05.tif to destination /media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/20220517_SASMDD0012_dual_ortho_01_bilinear.tif.
# ....1010.....20....2030.....40.30...50......6040....70....50.80.....90...60..100 - done.
# Total execution time: 1665.03 seconds
# Rasterization completed successfully.
