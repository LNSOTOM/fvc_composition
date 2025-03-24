
#%%
import geopandas as gpd
import pandas as pd
import os

def merge_shapefiles(folder_path):
    # List all files in the folder
    shapefiles = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.shp')]
    
    # Check if there are any shapefiles in the folder
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles found in the folder: {folder_path}")
    
    # Load and merge all shapefiles into a single GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(shp) for shp in shapefiles], ignore_index=True))
    
    return merged_gdf


## low
# folder_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'
## medium
# folder_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'  # Update this path to your folder
## dense
folder_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'
merged_shapefiles = merge_shapefiles(folder_path)


## Save the merged GeoDataFrame to a new shapefile low
# output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/merged/fvc_class_SASMDD0011.shp'
## Save the merged GeoDataFrame to a new shapefile medium
# output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/merged/fvc_class_SASMDD0001.shp'
## Save the merged GeoDataFrame to a new shapefile dense
output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/merged/fvc_class_SASMDD0012.shp'
merged_shapefiles.to_file(output_path)

print(f"Merged shapefiles saved to: {output_path}")


# %%
import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np

def merge_rasters(folder_path, output_path):
    # List all GeoTIFF files in the folder
    raster_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    # Check if there are any raster files in the folder
    if not raster_files:
        raise FileNotFoundError(f"No raster files found in the folder: {folder_path}")
    
    # Open all raster files as datasets
    datasets = [rasterio.open(raster) for raster in raster_files]
    
    # Merge all raster files
    merged_array, merged_transform = merge(datasets)
    
    # Copy metadata from one of the input rasters
    out_meta = datasets[0].meta.copy()
    
    # Update metadata with new dimensions and transform
    out_meta.update({
        "driver": "GTiff",
        "height": merged_array.shape[1],
        "width": merged_array.shape[2],
        "transform": merged_transform
    })
    
    # Save the merged raster to the output path
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(merged_array)
    
    # Close all datasets
    for dataset in datasets:
        dataset.close()
    
    print(f"Merged raster saved to: {output_path}")

# Folder containing raster files
## low
folder_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster'
## medium
# folder_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_rasterd'
## dense
# folder_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster'

# Output raster file path
## low
output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/merged/fvc_class_SASMDD0011.tif'
## medium
# output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/merged/fvc_class_SASMDD0001.tif'
## dense
# output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/raster_files/merged/fvc_class_SASMDD0012.tif'

# Call the merge_rasters function
merge_rasters(folder_path, output_path)


###teste 2
# %%
import os
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd

def merge_and_crop_rasters(raster_folder, shapefile_path, output_path):
    """
    Merge all rasters in a folder and crop the merged raster using a shapefile as a boundary,
    keeping existing NaN (no-data) values intact.

    Parameters:
    raster_folder (str): Path to the folder containing raster files.
    shapefile_path (str): Path to the shapefile used as a boundary.
    output_path (str): Path to save the cropped, merged raster.
    """
    # Load the shapefile
    shapefile_gdf = gpd.read_file(shapefile_path)
    
    # Ensure the shapefile CRS matches the raster CRS
    shapefile_crs = shapefile_gdf.crs
    
    # List all raster files in the folder
    raster_files = [os.path.join(raster_folder, f) for f in os.listdir(raster_folder) if f.endswith('.tif')]
    if not raster_files:
        raise FileNotFoundError(f"No raster files found in the folder: {raster_folder}")
    
    # Open all rasters as datasets
    datasets = [rasterio.open(raster_file) for raster_file in raster_files]
    
    # Merge all rasters
    merged_array, merged_transform = merge(datasets)
    


#%%
import os
from osgeo import gdal, ogr

def merge_and_crop_rasters_gdal(raster_folder, shapefile_path, output_path):
    """
    Merge raster files in a folder and crop the merged raster to the extent of a shapefile using GDAL.

    Parameters:
    raster_folder (str): Path to the folder containing raster files.
    shapefile_path (str): Path to the shapefile used as a boundary.
    output_path (str): Path to save the cropped, merged raster.
    """
    # List all raster files in the folder
    raster_files = [os.path.join(raster_folder, f) for f in os.listdir(raster_folder) if f.endswith('.tif')]
    if not raster_files:
        raise FileNotFoundError(f"No raster files found in the folder: {raster_folder}")
    
    # Create a temporary merged raster file
    merged_raster_path = "temp_merged.tif"
    merge_options = gdal.WarpOptions(
        format="GTiff",
        options=["COMPRESS=LZW"]
    )
    gdal.Warp(merged_raster_path, raster_files, options=merge_options)
    print(f"Merged raster saved temporarily at: {merged_raster_path}")
    
    # Open the shapefile and get its extent
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()
    extent = layer.GetExtent()  # Returns (minX, maxX, minY, maxY)

    # Use gdal.Warp to crop the merged raster to the shapefile extent
    crop_options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(extent[0], extent[2], extent[1], extent[3]),  # minX, minY, maxX, maxY
        cutlineDSName=shapefile_path,  # Shapefile for exact cropping
        cropToCutline=True,
        dstNodata=None,  # Preserve existing NoData values
        options=["COMPRESS=LZW"]
    )
    gdal.Warp(output_path, merged_raster_path, options=crop_options)
    print(f"Merged and cropped raster saved to: {output_path}")
    
    # Clean up temporary merged raster
    if os.path.exists(merged_raster_path):
        os.remove(merged_raster_path)
        print("Temporary merged raster file deleted.")

# Inputs
## low
raster_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster'
shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/merged/fvc_class_SASMDD0011.shp'
output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/merged/fvc_class_SASMDD0011.tif'

## dense
# raster_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/raster_files'
# shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/reference.shp'
# output_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/raster_files/merged/fvc_class_SASMDD0012.tif'

# Call the function
merge_and_crop_rasters_gdal(raster_folder, shapefile_path, output_path)

# %%
import geopandas as gpd

def main():
    input_shp = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/merged/fvc_class_SASMDD0001.shp'
    
    gdf = gpd.read_file(input_shp)
    
    # Print all available column names
    print("Available columns:", gdf.columns.tolist())
    
    # Continue with your filtering once you verify the correct column names.
    # For example, if the field is actually named 'branch_stem' instead of 'branch_stem_standing',
    # update your filter accordingly.
    
if __name__ == "__main__":
    main()

#%%
import geopandas as gpd

def main():
    # Path to the input shapefile (update with your file path)    
    input_shp = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/merged/fvc_class_SASMDD0001.shp'
    
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(input_shp)
    
    # Print available columns for debugging purposes
    print("Available columns:", gdf.columns.tolist())
    
    # Filter rows where the "class" attribute is "NPV"
    npv_gdf = gdf[gdf["class"] == "npv"]
    
    # Define the list of valid structure attributes
    valid_structures = ["branch_stem_standing", "cwd", "litter"]
    
    # Further filter rows where the "structure" attribute matches one of the valid options
    filtered_gdf = npv_gdf[npv_gdf["structure"].isin(valid_structures)]
    
    # Optional: Write the filtered data to a new shapefile
    output_shp = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/merged/npv_standingwood.shp"
    filtered_gdf.to_file(output_shp)
    print(f"Filtered shapefile has been saved to: {output_shp}")

if __name__ == "__main__":
    main()

# %%
