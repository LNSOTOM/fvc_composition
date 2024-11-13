

#%%
#STEP 1 (option a)
# Step 1_2:  Convert a single TIFF file to Cloud Optimized GeoTIFF (COG) format (aimed at being hosted on a HTTP file server)
import subprocess

#site1 [MEDIUM]
# input_tif = r'/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/orthomosaic/20220519_SASMDD001_dual_ortho_01_bilinear.tif'
# output_tif = r'/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/orthomosaic/cog/20220519_SASMDD001_dual_ortho_01_bilinear_cog.tif'

#site 11 [LOW]
# input_tif = r'/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_01_bilinear.tif'
# output_tif = r'/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/cog/20220517_SASMDD0011_dual_ortho_01_bilinear_cog.tif'

#site 12 [DENSE]
input_tif = r'/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/20220517_SASMDD0012_dual_ortho_01_bilinear.tif'
output_tif = r'/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/cog/20220517_SASMDD0012_dual_ortho_01_bilinear.tif_cog.tif'

# Construct the gdal_translate command
gdal_translate_cmd = [
    "gdal_translate",
    "-of", "COG",
    "-ot", "Float32",
    "-co", "COMPRESS=DEFLATE",
    "-co", "PREDICTOR=2",
    "-co", "BIGTIFF=YES",
    "-a_nodata", "-32767",
    input_tif,
    output_tif
]

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_tif), exist_ok=True)

# Run the gdal_translate command
try:
    subprocess.run(gdal_translate_cmd, check=True)
    print("Conversion to COG successful!")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")


#%%
## PHASE 1_Image Processing
###### THIS GOES FASTER
#STEP 1 (option b)
# Step 1_2:  Convert a single TIFF file to Cloud Optimized GeoTIFF (COG) format (aimed at being hosted on a HTTP file server)
import subprocess
import concurrent.futures
import os


# List of input TIFF files and their corresponding output COG file names
files = [
    # {
    #     #site 1
    #     'input': '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/orthomosaic/20220519_SASMDD001_dual_ortho_01_bilinear.tif',
    #     'output': '/media/laura/Extreme SSD/qgis/calperumResearch/site1_supersite_DD0001/orthomosaic/20220519_SASMDD001_dual_ortho_01_bilinear_cog.tif'
    # },
    {
        #site 11 
        'input': '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_01_bilinear.tif',
        'output': '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_01_bilinear_cog.tif'
    },
    {
        #site 12
        'input': '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/20220517_SASMDD0012_dual_ortho_01_bilinear.tif',
        'output': '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/20220517_SASMDD0012_dual_ortho_01_bilinear.tif_cog.tif'
    }
    # Add more files as needed
]

def convert_to_cog(file_info):
    """
    Convert a single TIFF file to COG format using gdal_translate.
    """
    input_tif = file_info["input"]
    output_tif = file_info["output"]
    
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    
    gdal_translate_cmd = [
        "gdal_translate",
        "-of", "COG",
        "-ot", "Float32",
        "-co", "COMPRESS=DEFLATE",  #LZW
        "-co", "PREDICTOR=2",  
        "-co", "BIGTIFF=YES",
        "-a_nodata", "-32767",
        input_tif, output_tif
    ]

    try:
        subprocess.run(gdal_translate_cmd, check=True)
        print(f"Conversion to COG successful for {input_tif}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_tif}: {e}")

# Use ThreadPoolExecutor to convert files in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Map the convert_to_cog function to the files list
    results = list(executor.map(convert_to_cog, files))

# Print the result of each conversion
for result in results:
    print(result)

