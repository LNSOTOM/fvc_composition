## PHASE 2_Reference data
#STEP 2
# Step 2_1: Convert JSON (from LABELME) to vector data for QGIS filter (shpFormat)
########### option c) with extra label description
import os
import json
import rasterio
from shapely.geometry import Polygon
import geopandas as gpd

def pixel_to_geo(points, transform):
    """Convert pixel coordinates to geographic coordinates using the affine transform."""
    return [(transform * point)[:2] for point in points]

def json_to_shapefile(json_path, composite_path, shapefile_path):
    base_filename = os.path.splitext(os.path.basename(composite_path))[0]
    output_filename = f"pv_class_{base_filename}.shp"
    output_path = os.path.join(shapefile_path, output_filename)
    os.makedirs(shapefile_path, exist_ok=True)
    
    with open(json_path, 'r') as file:
        data = json.load(file)

    with rasterio.open(composite_path) as src:
        raster_crs = src.crs
        transform = src.transform

    polygons = []
    labels = []
    descriptions = []  # Now directly using descriptions from the JSON
    group_ids = []
    centroids_x = []
    centroids_y = []

    for feature in data['shapes']:
        if feature['shape_type'] == 'polygon':
            geo_points = pixel_to_geo(feature['points'], transform)
            polygon = Polygon(geo_points)
            polygons.append(polygon)
            labels.append(feature['label'])
            descriptions.append(feature.get('description', "No description"))  # Use the description if available
            group_ids.append(feature.get('group_id', None))
            centroid = polygon.centroid
            centroids_x.append(centroid.x)
            centroids_y.append(centroid.y)

    gdf = gpd.GeoDataFrame({
        'geometry': polygons,
        'id': group_ids,
        'class': labels,
        'structure': descriptions,  #type
        'x': centroids_x,
        'y': centroids_y
    }, crs=raster_crs)

    # Ensuring data types
    gdf['id'] = gdf['id'].astype('Int64')
    gdf['class'] = gdf['class'].astype(str).str.slice(0, 50)  # Apply limit to string length if necessary
    gdf['structure'] = gdf['structure'].astype(str).str.slice(0, 128)  # Apply limit to string length if necessary
    gdf['x'] = gdf['x'].astype(float)
    gdf['y'] = gdf['y'].astype(float)
    
    gdf.to_file(output_path)


# file paths
# composite_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.22.tif'
composite_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/composite_colour_raster_3b/composite_percentile_tiles_multispectral.48.tif'
# rgb_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/tiles_rgb/tiles_rgb_3072.17.tif'
# rgb_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/tiles_rgb/tiles_rgb_3072.16.tif'


# json_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_json/composite_percentile_tiles_multispectral.22.json'
json_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_json/composite_percentile_tiles_multispectral.48.json'

# json_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_json/tiles_rgb_3072.17.json'
# json_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_json/tiles_rgb_3072.16.json'


# shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/extra_description'
shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp'
# shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/tiles_rgb_3072.17'
# shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/tiles_rgb_3072.16'



json_to_shapefile(json_path, composite_path, shapefile_path)

# %%
########## plot
# import geopandas as gpd
# import matplotlib.pyplot as plt

# # Assuming shapefile_path is the directory where the shapefile is saved
# # and output_filename is the name of the shapefile created from JSON annotations
# output_filename = 'globStat_percentile_tiles_rgb_fobs.125_annotated.shp'
# # output_filename = 'globStat_percentile_tiles_rgb_fobs.125_poly_annotated.shp'
# shapefile_full_path = os.path.join(shapefile_path, output_filename)

# # Read the shapefile
# gdf = gpd.read_file(shapefile_full_path)

# # Plot the shapefile
# fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size as needed
# gdf.plot(ax=ax)

# # Setting plot titles and labels, if needed
# ax.set_title('Shapefile Plot')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')

# plt.show()


# # %%
# # plot_labelme_shp_outputs
# import geopandas as gpd


# # shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/glob_stats/tiles_rgb_large_2048/pv_class/globStat_percentile_tiles_rgb_fobs.125_poly_annotated.shp'
# shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/annotation_shp/pv_class_composite_percentile_tiles_multispectral.22.shp'



# annotated_shp = gpd.read_file(shapefile_path)
# annotated_shp.head()

# check database dimensions
# print('Examples:{}\nFeatures: {}'.format(annotated_shp.shape[0], annotated_shp.shape[1]))