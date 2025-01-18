
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
