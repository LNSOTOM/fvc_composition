## PHASE 1_Image Processing
#STEP 4
# Step 4_2: Apply percentile stretch to MULTISPECTRAL
################ FOR MUTLIPLE FILES FOR MULTISPECTRAL (option a)
import os
import numpy as np
import rasterio
import json


#site1
stats_json_path = '/home/laura/Documents/code/ecosystem_composition/phase_1_image_processing/step_5_percentile_normalisedPixels_model/global_statistics/site1_DD0001_statistics_multispectral.json'
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
#### Larger 3072x3072
#site1 - supersite [MEDIUM]
##sample 30
input_folder  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral'
#extra 5 for 35 sample
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/random_sample35/tiles_multispectral'

output_folder  = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/global_stats/tiles_multispectral'

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
                

#%%
########## Dask parallel distribution (in trial-option b)
from dask.distributed import Client, performance_report
from dask import delayed, compute
import numpy as np
import dask.array as da
import rasterio
import time
import json
import webbrowser

# Initialize Dask client with specified dashboard address
client = Client(processes=False, dashboard_address=':5055')

# Set this flag to True to wait for the user to view the Dask dashboard
WAIT_DASHBOARD = True

if WAIT_DASHBOARD:
    print("Dask client information:", client)
    print("Open browser and enter the following URL for Dask dashboard: " + str(client.dashboard_link))
    webbrowser.open(client.dashboard_link, new=2)
    input("Press Enter to continue after you have inspected the Dask dashboard...")

def compute_band_statistics(image_path, band_number, chunk_size=1000):
    with rasterio.open(image_path) as src:
        band = src.read(band_number)
        # Convert the band data to a Dask array
        dask_band = da.from_array(band, chunks=(chunk_size, chunk_size))
        
        # Compute statistics using Dask
        min_value = da.nanmin(dask_band).compute()
        max_value = da.nanmax(dask_band).compute()
        mean_value = da.nanmean(dask_band).compute()
        std_value = da.nanstd(dask_band).compute()
        
        return {
            'min': min_value,
            'max': max_value,
            'mean': mean_value,
            'std': std_value
        }
        
def compute_global_statistics_dask(image_path):
    with rasterio.open(image_path) as ds:
        stats = [compute_band_statistics(image_path, band_num+1) for band_num in range(ds.count)]
        
    return stats


# Path to first raster file
#site1 - supersite [MEDIUM]
first_image_path  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_supersite_DD0001/orthomosaic/20220519_SASMDD001_dual_ortho_01_bilinear.tif'

#site2 - similar to supersite [MEDIUM]
# first_image_path  = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/orthomosaic/20220518_SASMDD0010_18_dual_ortho_01_bilinear.tif'

#site3 - with water  [DENSE]
# first_image_path  = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/20220517_SASMDD0012_dual_ortho_01_bilinear.tif'


# Start timing
t0 = time.perf_counter()

# Use the performance report to monitor execution and generate a report
with performance_report(filename="dask_performance_report.html"):
    first_image_stats = compute_global_statistics_dask(first_image_path)

overall_statistics = {
    'overall_mean': np.mean([stat['mean'] for stat in first_image_stats]),
    'overall_std': np.mean([stat['std'] for stat in first_image_stats]),
    'overall_min': min(stat['min'] for stat in first_image_stats),
    'overall_max': max(stat['max'] for stat in first_image_stats),
}

print(f"Overall Statistics: Mean={overall_statistics['overall_mean']:.2f}, Std={overall_statistics['overall_std']:.2f}, Min={overall_statistics['overall_min']:.2f}, Max={overall_statistics['overall_max']:.2f}")

output_filename = 'site1_DD0001_statistics_multispectral_dask.json'
with open(output_filename, 'w') as outfile:
    json.dump({
        'band_statistics': first_image_stats,
        'overall_statistics': overall_statistics
    }, outfile, indent=4)

print(f"Statistics saved to {output_filename}")

# Stop timing
t1 = time.perf_counter()
tdiff = t1 - t0
print(f"It took {tdiff:.3f} seconds to process the raster bands and calculate statistics")