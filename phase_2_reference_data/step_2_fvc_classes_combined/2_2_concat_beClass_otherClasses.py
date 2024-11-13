## PHASE 2_Reference data
#STEP 2
# Step 2_2: Concatenate step1_1_1_be class with step2_2_1_classes vector data
########### improve concatennate to crop pv class combined with be class (good version)
## option a)
import geopandas as gpd
import pandas as pd


# Paths to your shapefiles
## low site
shapefile_path_1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/mask_be_globStat_percentile_tiles_multispectral.79.shp'
shapefile_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/pv_class_composite_percentile_tiles_multispectral.79.shp'

## medium site
# shapefile_path_1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/mask_be_globStat_percentile_tiles_multispectral.22.shp'
# shapefile_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/pv_class_composite_percentile_tiles_multispectral.22.shp'

## dense site
# shapefile_path_1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/new/mask_be_globStat_percentile_tiles_multispectral.128.shp'
# shapefile_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/pv_class/annotation_shp/pv_class_composite_percentile_tiles_multispectral.128.shp'
# # Output path for the result
## low site
# output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.79.shp'

# ## medium site
output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.22.shp'

# dense site
# output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.128.shp'


# Load the shapefiles as GeoDataFrames
gdf1 = gpd.read_file(shapefile_path_1)
gdf2 = gpd.read_file(shapefile_path_2)

# Make sure both GeoDataFrames use the same CRS
gdf2 = gdf2.to_crs(gdf1.crs)

# 1. Union: This step is optional based on the explanation but necessary if you want to identify all unique areas.
# It's included here for completeness and potential use in visualization or further analysis.
# Perform a union operation without dropping any geometries due to type differences
union_gdf = gpd.overlay(gdf1, gdf2, how='union', keep_geom_type=False)

# 2. Difference: Remove parts of gdf1 that intersect with gdf2
difference_gdf = gpd.overlay(gdf1, gdf2, how='difference')

# Now, you might want to combine (concatenate) the non-intersecting parts of gdf1 with all of gdf2
# To do this, simply concatenate the difference_gdf with gdf2
result_gdf = gpd.GeoDataFrame(pd.concat([difference_gdf, gdf2], ignore_index=True))

# Save the result to a new shapefile
result_gdf.to_file(output_shapefile_path)

print(f"Modified shapefile saved to {output_shapefile_path}")


#%%
### option b) run this if fix TopologyException side location conflict.
import geopandas as gpd
import pandas as pd

## low site
# shapefile_path_1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/mask_be_globStat_percentile_tiles_multispectral.66.shp'
# shapefile_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/pv_class_composite_percentile_tiles_multispectral.66.shp'

## medium site
# shapefile_path_1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/mask_be_globStat_percentile_tiles_multispectral.93.shp'
# shapefile_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/pv_class_composite_percentile_tiles_multispectral.93.shp'

## dense site
shapefile_path_1 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/mask_be_globStat_percentile_tiles_multispectral.53.shp'
shapefile_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/pv_class/annotation_shp/pv_class_composite_percentile_tiles_multispectral.53.shp'

# Output path for the result
## low site
# output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.66.shp'

## medium site
# output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.93.shp'

## dense site
output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.53.shp'


# Load the shapefiles as GeoDataFrames
gdf1 = gpd.read_file(shapefile_path_1)
gdf2 = gpd.read_file(shapefile_path_2)

# Make sure both GeoDataFrames use the same CRS
gdf2 = gdf2.to_crs(gdf1.crs)

# Check for invalid geometries and fix them
gdf1['geometry'] = gdf1['geometry'].buffer(0)
gdf2['geometry'] = gdf2['geometry'].buffer(0)

# Simplify geometries if needed
gdf1['geometry'] = gdf1['geometry'].simplify(tolerance=0.001)
gdf2['geometry'] = gdf2['geometry'].simplify(tolerance=0.001)

# 1. Union: Perform a union operation without dropping any geometries due to type differences
union_gdf = gpd.overlay(gdf1, gdf2, how='union', keep_geom_type=False)

# 2. Difference: Remove parts of gdf1 that intersect with gdf2
difference_gdf = gpd.overlay(gdf1, gdf2, how='difference')

# Combine (concatenate) the non-intersecting parts of gdf1 with all of gdf2
result_gdf = gpd.GeoDataFrame(pd.concat([difference_gdf, gdf2], ignore_index=True))

# Save the result to a new shapefile
result_gdf.to_file(output_shapefile_path)

print(f"Modified shapefile saved to {output_shapefile_path}")


#%%
### upodate coord
import geopandas as gpd
import pandas as pd

# Load the input shapefile into a GeoDataFrame
#low site
# input_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.16.shp'
#medium site
input_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.22.shp'
#dense site
# input_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.128.shp'


gdf = gpd.read_file(input_shapefile_path)

# Check for missing 'x' or 'y' values and compute centroids for these cases
for index, row in gdf.iterrows():
    if pd.isnull(row['x']) or pd.isnull(row['y']):
        # Ensure the geometry is not None before computing the centroid
        if row.geometry is not None:
            centroid = row.geometry.centroid
            gdf.at[index, 'x'] = centroid.x
            gdf.at[index, 'y'] = centroid.y
        else:
            # Handle None geometries here (e.g., log a warning or fill with a default value)
            print(f"Warning: Row {index} has a None geometry.")

# Save the updated GeoDataFrame back to the shapefile (or a new one if you prefer)
#low site
# output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.16.shp'
#medium site
output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.22.shp'
# dense site
# output_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/mask_fvc_3072.128.shp'



gdf.to_file(output_shapefile_path)

print("Updated shapefile with missing centroid coordinates.")

# %%
### add area
#### calculate area
import geopandas as gpd
import os

# low site
# directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/'
# output_directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'

# medium site
directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/'
output_directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'

# dense site
# directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/'
# output_directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'



# Ensure the output directory exists
os.makedirs(output_directory_path, exist_ok=True)

# Ensure the output directory exists
os.makedirs(output_directory_path, exist_ok=True)

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a shapefile
    if filename.endswith(".shp"):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        print(f"Processing shapefile: {file_path}")
        
        # Load the shapefile as a GeoDataFrame
        gdf = gpd.read_file(file_path)

        # Calculate the area of each polygon and add it as a new column
        # Explicitly set the data type of the 'area' column to float
        gdf['area'] = gdf.geometry.area.astype(float)

        # Construct the output file path
        output_file_path = os.path.join(output_directory_path, filename)
        
        # Save the updated GeoDataFrame back to a new shapefile
        gdf.to_file(output_file_path)

        print(f"Updated shapefile saved to: {output_file_path}")
        
# %%
####### Rasterised as mask based on calss features - shp to raster for mask
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
import os
import numpy as np


def shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path):
    """
    Convert a shapefile to a raster based on class attributes,
    using a reference raster for spatial extent and resolution, with specified profile settings.
    """
    shapes = gpd.read_file(shapefile_path)
    class_mapping = {'be': 0, 'npv': 1, 'pv': 2, 'si': 3, 'wi': 4}
    shapes['class_value'] = shapes['class'].map(class_mapping)
    
    with rasterio.open(reference_raster_path) as ref:
        # Define the output profile based on the reference raster and additional specifications
        profile = ref.profile.copy()
        profile.update(
            driver='GTiff',
            count=1,
            compress='lzw',
            nodata=None,
            dtype=rasterio.float32,
            tiled=True,
            blockxsize=3072,
            blockysize=3072
        )

        rasterized = rasterize(
            ((geometry, value) for geometry, value in zip(shapes.geometry, shapes.class_value)),
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=np.nan,  # Assuming you wish to use np.nan for areas not covered by any polygon
            all_touched=True,
            dtype=rasterio.float32
        )

        # Write the rasterized output with the updated profile
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(rasterized, 1)

# low site
# shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.66.shp'
# output_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.66.tif'
# reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral/tiles_multispectral.66.tif'

# medium site
shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.22.shp'
output_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.22.tif'
reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral/tiles_multispectral.22.tif'

# dense site
# shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.128.shp'
# output_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.128.tif'
# reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/raw/tiles_multispectral/tiles_multispectral.128.tif'


shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path)


#%% ####### Rasterised as mask based on be class features - shp to raster for mask (optional)
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
import os
import numpy as np


def shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path):
    """
    Convert a shapefile to a raster based on class attributes,
    using a reference raster for spatial extent and resolution, with specified profile settings.
    """
    shapes = gpd.read_file(shapefile_path)
    class_mapping = {'be': 0, 'npv': 1, 'pv': 2, 'si': 3, 'wi': 4}
    shapes['class_value'] = shapes['class'].map(class_mapping)
    
    with rasterio.open(reference_raster_path) as ref:
        # Define the output profile based on the reference raster and additional specifications
        profile = ref.profile.copy()
        profile.update(
            driver='GTiff',
            count=1,
            compress='lzw',
            nodata=None,
            dtype=rasterio.float32,
            tiled=True,
            blockxsize=3072,
            blockysize=3072
        )

        rasterized = rasterize(
            ((geometry, value) for geometry, value in zip(shapes.geometry, shapes.class_value)),
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=np.nan,  # Assuming you wish to use np.nan for areas not covered by any polygon
            all_touched=True,
            dtype=rasterio.float32
        )

        # Write the rasterized output with the updated profile
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(rasterized, 1)

# low site
# shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.66.shp'
# output_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.66.tif'
# reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral/tiles_multispectral.66.tif'

# medium site
shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/annotation_shp/mask_be_globStat_percentile_tiles_multispectral.22.shp'
output_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/glob_stats/be_class/be_mask_threshold.22.tif'
reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral/tiles_multispectral.22.tif'

# dense site
# shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.128.shp'
# output_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.128.tif'
# reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/raw/tiles_multispectral/tiles_multispectral.128.tif'


shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path)