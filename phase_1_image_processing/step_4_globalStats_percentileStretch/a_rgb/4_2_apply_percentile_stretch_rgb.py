## PHASE 1_Image Processing
#STEP 4
# Step 4_2: Apply percentile stretch to RGB
################ FOR MUTLIPLE FILES FOR RGB
import os
import numpy as np
import rasterio
import json

#site1
stats_json_path = '/home/laura/Documents/code/ecosystem_composition/phase_1_image_processing/step_5_percentile_normalisedPixels_model/global_statistics/site1_DD0001_statistics_rgb.json'
with open(stats_json_path, 'r') as infile:
    data = json.load(infile)
first_image_stats = data['band_statistics']  # This should correctly reference the list of band statistics


def apply_percentile_stretch_to_band(band_data, stats):
    lower_percentile_value = stats['mean'] - 2 * stats['std']
    upper_percentile_value = stats['mean'] + 2 * stats['std']
    lower_percentile_value = max(lower_percentile_value, stats['min'])
    upper_percentile_value = min(upper_percentile_value, stats['max'])

    actual_lower_percentile = np.percentile(band_data, 1)
    actual_upper_percentile = np.percentile(band_data, 99)
    
    stretched_data = np.interp(band_data, 
                               [actual_lower_percentile, actual_upper_percentile], 
                               [lower_percentile_value, upper_percentile_value])
    
    return stretched_data


reference_image_stats = first_image_stats  # Assume this is already defined or loaded

# Folder for Random Sampling = 100 containing RGB TIFF images
### Small 256x256
#site1 - supersite [MEDIUM]
input_folder  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/tiles_rgb'
##extra 5 for 35 sample
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/random_sample35/tiles_rgb'

output_folder  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/global_stats/tiles_rgb'

#site2 - similar to supersite [MEDIUM]
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/predictors/tiles_multispectral'
# output_folder = '/home/laura/Documents/uas_data/Calperum/site2_DD0010_18/tiles_3840/global_stats/tiles_multispectral'

#site3 - with water  [DENSE]
# input_folder  = '/home/laura/Documents/uas_data/Calperum/site3_DD0012/tiles_3840/raw/tiles_multispectral'
# output_folder  = '/home/laura/Documents/uas_data/Calperum/site3_DD0012/tiles_3840/glob_stats/tiles_multispectral'


#### Larger 3072x3072
#site1 - supersite [MEDIUM]
##sample 30
# input_folder  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral'
##extra 5 for 35 sample
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/random_sample35/tiles_multispectral'
# output_folder  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/global_stats/tiles_multispectral'

#site2 - similar to supersite [MEDIUM]
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/predictors/tiles_multispectral'
# output_folder = '/home/laura/Documents/uas_data/Calperum/site2_DD0010_18/tiles_3840/global_stats/tiles_multispectral'

#site3 - with water  [DENSE]
# input_folder  = '/home/laura/Documents/uas_data/Calperum/site3_DD0012/tiles_3840/raw/tiles_multispectral'
# output_folder  = '/home/laura/Documents/uas_data/Calperum/site3_DD0012/tiles_3840/glob_stats/tiles_multispectral'


# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# List all the input raster files
input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

for filename in input_files:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f'globStat_percentile_{filename}')
    
    with rasterio.open(input_path) as src:
        meta = src.meta
        with rasterio.open(output_path, 'w', **meta) as dest:
            for band_index in range(1, src.count + 1):
                band_data = src.read(band_index)
                band_stats = reference_image_stats[band_index - 1]  # Adjust based on 0-indexing
                stretched_band = apply_percentile_stretch_to_band(band_data, band_stats)
                dest.write(stretched_band, band_index)