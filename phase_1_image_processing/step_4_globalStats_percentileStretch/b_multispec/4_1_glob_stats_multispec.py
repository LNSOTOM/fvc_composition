## PHASE 1_Image Processing
#STEP 4
# Step 4_1:Compute global statistics from MUTLTISPECTRAL orthomosaic imagery
############## Improve performance with gdal
from osgeo import gdal
import numpy as np
import time
import json


# Path to first raster file
#site1 - supersite [MEDIUM]
# first_image_path  = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_supersite_DD0001/orthomosaic/20220519_SASMDD001_dual_ortho_01_bilinear.tif'

#site2 - similar to supersite [MEDIUM]
# first_image_path  = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/orthomosaic/20220518_SASMDD0010_18_dual_ortho_01_bilinear.tif'

#site3 - with water  [DENSE]
first_image_path  = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/20220517_SASMDD0012_dual_ortho_01_bilinear.tif'


def compute_global_statistics(image_path):
    # Open the raster file
    ds = gdal.Open(first_image_path, 1)

    # List to store statistics of each band
    stats= []

    # Check if the dataset was successfully opened
    if ds is None:
        print("Failed to open the raster file.")
    else:
        print(f"Number of bands in the image: {ds.RasterCount}")  # Print the number of bands
        # Iterate over all bands in the dataset
        for band_num in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(band_num)
            # Compute statistics for the band
            band_stats = band.ComputeStatistics(0)
            # Store band statistics in the list
            stats.append({
                'mean': band_stats[2],
                'std': band_stats[3],
                'min': band_stats[0],
                'max': band_stats[1],
            })           
            # Print the statistics for the current band
            print(f"Band {band_num} Statistics: Mean={band_stats[2]:.2f}, StdDev={band_stats[3]:.2f}, Min={band_stats[0]:.2f}, Max={band_stats[1]:.2f}")
          
        # Close the dataset
        ds = None
        
    return stats


# Start timing
start_time = time.time()

first_image_stats = compute_global_statistics(first_image_path)

if first_image_stats:  # Ensure there are stats to process
    # Calculate overall statistics across bands
    overall_statistics = {
        'overall_mean': np.mean([stat['mean'] for stat in first_image_stats]),
        'overall_std': np.mean([stat['std'] for stat in first_image_stats]),
        'overall_min': min(stat['min'] for stat in first_image_stats),
        'overall_max': max(stat['max'] for stat in first_image_stats),
    }

    # Print overall statistics
    print(f"Overall Statistics: Mean={overall_statistics['overall_mean']:.2f}, Std={overall_statistics['overall_std']:.2f}, Min={overall_statistics['overall_min']:.2f}, Max={overall_statistics['overall_max']:.2f}")

    # Combine band and overall statistics in a dictionary
    output_data = {
        'band_statistics': first_image_stats,
        'overall_statistics': overall_statistics
    }

    # Save the statistics to a JSON file
    # with open('site1_DD0001_statistics_multispectral.json', 'w') as outfile:
    #     json.dump(output_data, outfile, indent=4)
    # with open('site2_DD0010_18_statistics_multispectral.json', 'w') as outfile:
    #     json.dump(output_data, outfile, indent=4)       
    with open('site3_DD0012_statistics_mutlispectral.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print("Statistics saved to raster_statistics.json")
        
# Stop timing and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution Time: {elapsed_time:.2f} seconds")


#site1 - supersite [MEDIUM]
# Band 1 Statistics: Mean=0.06, StdDev=0.02, Min=0.00, Max=0.37
# Band 2 Statistics: Mean=0.07, StdDev=0.03, Min=0.00, Max=0.30
# Band 3 Statistics: Mean=0.09, StdDev=0.04, Min=0.00, Max=0.36
# Band 4 Statistics: Mean=0.11, StdDev=0.05, Min=0.00, Max=0.39
# Band 5 Statistics: Mean=0.18, StdDev=0.10, Min=0.00, Max=0.60
# Band 6 Statistics: Mean=0.18, StdDev=0.10, Min=0.00, Max=0.61
# Band 7 Statistics: Mean=0.21, StdDev=0.10, Min=0.00, Max=0.68
# Band 8 Statistics: Mean=0.22, StdDev=0.10, Min=0.00, Max=0.68
# Band 9 Statistics: Mean=0.24, StdDev=0.10, Min=0.00, Max=0.69
# Band 10 Statistics: Mean=0.27, StdDev=0.10, Min=0.01, Max=0.77
# Overall Statistics: Mean=0.16, Std=0.08, Min=0.00, Max=0.77
# Statistics saved to raster_statistics.json
# Execution Time: 3157.33 seconds

#site2 - similar to supersite [MEDIUM]
# Number of bands in the image: 10
# Band 1 Statistics: Mean=0.05, StdDev=0.02, Min=0.00, Max=0.37
# Band 2 Statistics: Mean=0.06, StdDev=0.02, Min=0.00, Max=0.42
# Band 3 Statistics: Mean=0.08, StdDev=0.03, Min=0.01, Max=0.54
# Band 4 Statistics: Mean=0.10, StdDev=0.05, Min=0.01, Max=0.62
# Band 5 Statistics: Mean=0.17, StdDev=0.10, Min=0.00, Max=1.00
# Band 6 Statistics: Mean=0.18, StdDev=0.10, Min=0.01, Max=0.90
# Band 7 Statistics: Mean=0.21, StdDev=0.11, Min=0.01, Max=0.95
# Band 8 Statistics: Mean=0.22, StdDev=0.10, Min=0.01, Max=0.96
# Band 9 Statistics: Mean=0.25, StdDev=0.11, Min=0.02, Max=0.99
# Band 10 Statistics: Mean=0.29, StdDev=0.11, Min=0.03, Max=1.00
# Overall Statistics: Mean=0.16, Std=0.08, Min=0.00, Max=1.00
# Statistics saved to raster_statistics.json
# Execution Time: 3216.65 seconds

#site3 - with water  [DENSE]
# Band 1 Statistics: Mean=127.54, StdDev=64.31, Min=0.00, Max=255.00
# Band 2 Statistics: Mean=119.60, StdDev=64.94, Min=0.00, Max=255.00
# Band 3 Statistics: Mean=101.39, StdDev=68.05, Min=0.00, Max=255.00
# Overall Statistics: Mean=116.18, Std=65.77, Min=0.00, Max=255.00
# Statistics saved to raster_statistics.json
# Execution Time: 13.16 seconds