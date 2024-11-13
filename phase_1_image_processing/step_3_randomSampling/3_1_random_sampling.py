#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:07:13 2024

@author: lauransotomayor
"""
## PHASE 1_Image Processing
#STEP 3: Random sampling tiles (up to 35 tiles) - Matching RGB and MULTISPECTRAL location
#################### Code improved
''' to find matches for as many multispectral images as possible until the desired number of RGB matches is found,
potentially iterating through more multispectral images than the initial target number to compensate 
for RGB images excluded due to white pixels.
'''
import numpy as np
import os
import random
import shutil
import tifffile
import time


start_time = time.time()  # Start timing


def contains_white_pixels(image_path):
    img_array = tifffile.imread(image_path)
    # Ensure the image is in the expected format (RGB)
    if img_array.ndim == 3 and img_array.shape[-1] == 3:
        # Check for white pixels (all channels at 255)
        return np.any(np.all(img_array == 255, axis=-1))
    else:
        # For non-RGB images or unexpected formats, return False or handle differently
        return False  # Assuming non-RGB images don't need this white pixel check

def extract_number_from_filename(filename):
    basename = os.path.splitext(filename)[0]
    number_part = basename.split('.')[-1]
    try:
        return int(number_part)
    except ValueError:
        return None

def match_and_copy_files(selected_images, source_folder, destination_folder):  
    # Sort selected images numerically based on the extracted number from their filenames
    selected_images_sorted = sorted(selected_images, key=lambda x: extract_number_from_filename(x))
    for image_name in selected_images_sorted:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copyfile(source_path, destination_path)
        print(f"Copying {image_name} to {destination_folder}")

def create_matched_training_data(source_folders, destination_folders, target_num_samples):
    rgb_source_folder, multispectral_source_folder = source_folders
    rgb_destination_folder, multispectral_destination_folder = destination_folders

    # Ensure destination folders exist
    for destination_folder in destination_folders:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

    ms_files = [f for f in os.listdir(multispectral_source_folder) if f.lower().endswith(('.tiff', '.tif'))]
    selected_ms_images = random.sample(ms_files, len(ms_files))  # Sample all, but will limit later

    selected_rgb_images = []
    matched_ms_images = []

    for ms_image in selected_ms_images:
        if len(matched_ms_images) >= target_num_samples:
            break  # Stop if we have enough matches
        number = extract_number_from_filename(ms_image)
        for rgb_image in os.listdir(rgb_source_folder):
            rgb_image_path = os.path.join(rgb_source_folder, rgb_image)
            if extract_number_from_filename(rgb_image) == number and not contains_white_pixels(rgb_image_path):
                selected_rgb_images.append(rgb_image)
                matched_ms_images.append(ms_image)
                break  # Found a match, move to the next ms_image

    if len(selected_rgb_images) < target_num_samples:
        print(f"Warning: Only found {len(selected_rgb_images)} matching RGB images for {target_num_samples} requested samples.")

    # Copying the matched files
    match_and_copy_files(matched_ms_images, multispectral_source_folder, multispectral_destination_folder)
    print(f"Source folder: {multispectral_source_folder}")
    print(f"Destination folder: {multispectral_destination_folder}")
   
    match_and_copy_files(selected_rgb_images, rgb_source_folder, rgb_destination_folder)
    print(f"Source folder: {rgb_source_folder}")
    print(f"Destination folder: {rgb_destination_folder}")
    
   
#site1 - supersite [MEDIUM] DONE 100samples
### Larger chunk 3072x3072 =30m
source_folders = [
    '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/tiles_3072/tiles_rgb', 
    '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/tiles_3072/tiles_multispectral'
]
destination_folders = [
    '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/random_sample35/tiles_rgb', 
    '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/random_sample35/tiles_multispectral'
]


#site2 - similar to supersite [MEDIUM] DONE 100samples
### Larger chunk 3840x3840 =10m
# source_folders = [
#     '/home/laura/Documents/uas_data/Calperum/site3_DD0012/tiles_3840/raw/tiles_rgb', 
#     '/home/laura/Documents/uas_data/Calperum/site2_DD0010_18/tiles_3840/global_stats/tiles_multispectral'
# ]
# destination_folders = [
#     '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_3840/global_stats/tiles_rgb', 
#     '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_3840/global_stats/tiles_multispectral'
# ]


# # site3 - with water  [DENSE] DONE 100samples
#### Larger
# source_folders = [
#     '/home/laura/Documents/uas_data/Calperum/site3_DD0012/tiles_rgb', 
#     '/home/laura/Documents/uas_data/Calperum/site3_DD0012/tiles_multispectral'
# ]
# destination_folders = [
#     '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/predictors/tiles_rgb', 
#     '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/predictors/tiles_multispectral'
# ] 

    
#site4 -  Vegetation [LOW] (potentially site DD0013) DONE 100samples
# source_folders = [
#     '/home/laura/Documents/uas_data/Calperum/site4_DD0011/tiles_rgb', 
#     '/home/laura/Documents/uas_data/Calperum/site4_DD0011/tiles_multispectral'
# ]
# destination_folders = [
#     '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/predictors/tiles_rgb', 
#     '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/predictors/tiles_multispectral'
# ]


# case for application
num_samples = 35
create_matched_training_data(source_folders, destination_folders, num_samples)

end_time = time.time()  # Capture end time
duration = end_time - start_time  # Calculate duration
# print(f"Total execution time: {duration} seconds")


print(f"Total time: {time.time() - start_time} seconds")