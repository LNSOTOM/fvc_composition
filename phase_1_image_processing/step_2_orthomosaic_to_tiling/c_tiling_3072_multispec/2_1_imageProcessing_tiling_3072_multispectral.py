"""
Created on Fri Sep  8 14:46:42 2023

@author: lauransotomayor
"""
## PHASE 1_Image Processing
############## Modify FASTER MULTISPECTRAL
#STEP 2
# Step 2_1: Split orthomosaic raster to 3072x3072 px --> 10m x 10m (option a.1) for the wholke orthomosaic
# Tiling from top-left to right (by rows)- WITH COMPRESSION

import os
from osgeo import gdal
import concurrent.futures

# Define the function to create a single chunk of the image
def create_chunk(args):
    image_path, output_path, filename_prefix, x, y, counter = args
    chunk_size = 3072  # Define the size of each chunk
    output_filename = f"{filename_prefix}.{counter}.tif"
    output_file_path = os.path.join(output_path, output_filename)
    
    # Open the source image
    ds = gdal.Open(image_path)
    # Use gdal.Translate to extract the specified chunk without changing the resolution
    gdal.Translate(output_file_path, ds, srcWin=[x, y, chunk_size, chunk_size], format="GTiff")

# Prepare tasks for chunking a large image into 3072x3072 tiles
def prepare_chunking_tasks(image_path, output_path, filename_prefix):
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"Failed to open image: {image_path}")
        return []

    width, height = ds.RasterXSize, ds.RasterYSize
    chunk_size = 3072
    tasks = []

    counter = 0
    # Generate tasks for each chunk, starting from the top-left and moving row-wise to the right
    for y in range(0, height, chunk_size):  # Start from top, move down
        for x in range(0, width, chunk_size):  # Start from left, move right
            tasks.append((image_path, output_path, filename_prefix, x, y, counter))
            counter += 1  # Increment counter for unique filename suffix
    return tasks

# Main function to execute parallel chunking
def main(in_path, large_image_filename, out_path_chunked, output_filename_prefix):
    large_image_path = os.path.join(in_path, large_image_filename)
    
    tasks = prepare_chunking_tasks(large_image_path, out_path_chunked, output_filename_prefix)

    # Use concurrent processing to handle chunking tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(create_chunk, tasks)

if __name__ == "__main__":
    # Define paths and filenames   
    #site1 - supersite [MEDIUM] 
    #Multispectral imagery
    in_path = '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/orthomosaic'
    large_image_filename = '20220519_SASMDD001_dual_ortho_01_bilinear.tif'
    out_path_chunked = '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/tiles_3072/tiles_multispectral'

    #site2 - similar to supersite [MEDIUM]
    #Multispectral imagery
    # in_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/orthomosaic/'
    # large_image_filename = '20220518_SASMDD0010_18_dual_ortho_01_bilinear.tif'
    # out_path_chunked = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/tiles_3072/raw/tiles_multispectral'


    #site3 - with water  [DENSE]
    #Multispectral imagery
    # in_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/'
    # large_image_filename = '20220517_SASMDD0012_dual_ortho_01_bilinear.tif'
    # out_path_chunked = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/tiles_3072/raw/tiles_multispectral/'
    
    
    #site4 -  Vegetation [LOW] (potentially site DD0013)
    # #Multispectral imagery
    # in_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/'
    # input_filename = '20220517_SASMDD0011_dual_ortho_01_bilinear_clip.tif'
    # out_path = '/home/laura/Documents/uas_data/Calperum/site4_DD0011/tiles_large_3840/tiles_multispectral/'
    # output_filename_prefix = 'tiles_multispectral_3072'
    
    # output name files
    output_filename_prefix = 'tiles_multispectral'
    # output_filename_prefix = 'tiles_multispectral_3072'

    # Ensure the output directory exists
    if not os.path.exists(out_path_chunked):
        os.makedirs(out_path_chunked)

    # Execute the main function
    main(in_path, large_image_filename, out_path_chunked, output_filename_prefix)
    print("Chunking completed.")
    
####### (option a.2)
# improve for MULTISPECTRAL BANDS
import os
import concurrent.futures
from osgeo import gdal

def create_chunk(args):

    image_path, output_path, filename_prefix, x, y, counter, bands = args
    chunk_size = 3072  # Define the size of each chunk
    output_filename = f"{filename_prefix}.{counter}.tif"
    output_file_path = os.path.join(output_path, output_filename)
    
    # Open the source image
    ds = gdal.Open(image_path)
    # Prepare gdal.Translate options for handling multiple bands
    options = gdal.TranslateOptions(format="GTiff",
                                     srcWin=[x, y, chunk_size, chunk_size],
                                     bandList=bands)
    # Use gdal.Translate to extract the specified chunk with all bands
    gdal.Translate(output_file_path, ds, options=options)

def prepare_chunking_tasks(image_path, output_path, filename_prefix):
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"Failed to open image: {image_path}")
        return []

    width, height = ds.RasterXSize, ds.RasterYSize
    bands = [i + 1 for i in range(ds.RasterCount)]  # List of bands to include in each chunk
    chunk_size = 3072
    tasks = []

    counter = 0
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            tasks.append((image_path, output_path, filename_prefix, x, y, counter, bands))
            counter += 1
    return tasks

def main(in_path, large_image_filename, out_path_chunked, output_filename_prefix):
    large_image_path = os.path.join(in_path, large_image_filename)
    
    tasks = prepare_chunking_tasks(large_image_path, out_path_chunked, output_filename_prefix)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(create_chunk, tasks)

if __name__ == "__main__":
  
    # Define paths and filenames   
    #site1_1 - supersite [MEDIUM]  DONE
    #Multispectral imagery
    # in_path = '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/orthomosaic'
    # large_image_filename = '20220519_SASMDD001_dual_ortho_01_bilinear.tif'
    # out_path_chunked = '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/tiles_3072/tiles_multispectral'

    #site1_2 - similar to supersite [MEDIUM]
    #Multispectral imagery
    # in_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/orthomosaic/'
    # large_image_filename = '20220518_SASMDD0010_18_dual_ortho_01_bilinear.tif'
    # out_path_chunked = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/tiles_3072/tiles_multispectral'


    #site2_1 -  Vegetation [LOW] (potentially site DD0013) DONE
    # #Multispectral imagery
    # in_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/'
    # large_image_filename = '20220517_SASMDD0011_dual_ortho_01_bilinear.tif'
    # out_path_chunked = '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/tiles_3072/tiles_multispectral/'
    
    
    #site3_1 - with water  [DENSE]
    #Multispectral imagery
    in_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/'
    large_image_filename = '20220517_SASMDD0012_dual_ortho_01_bilinear.tif'
    out_path_chunked = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/tiles_3072/tiles_multispectral/'
    
    #site3_2 -similar to site 12 [DENSE] 
    # in_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site5_DD0008/orthomosaic/'
    # large_image_filename = '20220516_SASMDD0008_dual_ortho_01_bilinear.tif'
    # out_path_chunked = '/media/laura/Extreme SSD/qgis/calperumResearch/site5_DD0008/tiles_3072/tiles_multispectral'
         
    
    output_filename_prefix = 'tiles_multispectral'

    # Ensure the output directory exists
    if not os.path.exists(out_path_chunked):
        os.makedirs(out_path_chunked)

    # Execute the main function
    main(in_path, large_image_filename, out_path_chunked, output_filename_prefix)
    print("Chunking completed.")
