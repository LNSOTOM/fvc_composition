## PHASE 1_Image Processing
#STEP 5
# Step 5_1: Stack Bands to Create Composite Colour (3b from float to 8bit)
# improve previous code with fill nan values
import rasterio
import numpy as np
import os

# Define your input and output folders
## sample 30
input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral'
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/random_sample35/tiles_multispectral'
## extra 5
output_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b'

# Define the nodata values for input and output
nodata_value_input = -32767.0
nodata_value_uint8 = 0

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Function to normalize and convert a band using the 99th percentile
def normalize_convert_band(band, nodata_value_float32):
    # Replace nodata values with NaN for processing
    band[band == nodata_value_float32] = np.nan
    
    # Calculate the 1st and 99th percentiles while ignoring NaN values
    p1, p99 = np.nanpercentile(band, [1, 99])
    
    # Handle case where p1 == p99 to avoid division by zero
    if p1 == p99:
        band_scaled = np.full(band.shape, fill_value=128)  # Mid-point of 0-255
    else:
        # Normalize the band to the 0-255 range based on the 1st and 99th percentiles
        band_scaled = np.clip((band - p1) / (p99 - p1) * 254, 0, 254) + 1

    # Ensure all NaN values are replaced with nodata_value_uint8 after scaling
    band_scaled = np.where(np.isnan(band_scaled), nodata_value_uint8, band_scaled)

    return band_scaled.astype(np.uint8)


# Process each file in the input directory
for filename in os.listdir(input_folder):
    if not filename.endswith('.tif'):
        continue  # Skip non-TIF files

    input_path = os.path.join(input_folder, filename)
    output_filename = f"composite_percentile_{filename}"
    output_path = os.path.join(output_folder, output_filename)

    try:
        with rasterio.open(input_path) as src:
            if src.nodata is None:
                src.nodata = nodata_value_input

            if src.count < max([2, 6, 10]):
                print(f"Skipping {filename}: Insufficient number of bands.")
                continue

            # Normalize and reorder specified bands [2, 6, 10]
            normalized_band2 = normalize_convert_band(src.read(2), src.nodata)
            normalized_band6 = normalize_convert_band(src.read(6), src.nodata)
            normalized_band10 = normalize_convert_band(src.read(10), src.nodata)
            
            # save normalized bands 
            bands_uint8 = [normalized_band10, normalized_band6, normalized_band2]

            # Update the metadata for output
            out_meta = src.meta.copy()
            out_meta.update({
                'dtype': 'uint8',
                'count': 3,
                'nodata': nodata_value_uint8
            })

            # Write the processed bands to a new raster
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                for i, band in enumerate(bands_uint8, start=1):
                    dest.write(band, i)

            print(f"Processed and saved: {output_filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Completed processing all eligible files.")