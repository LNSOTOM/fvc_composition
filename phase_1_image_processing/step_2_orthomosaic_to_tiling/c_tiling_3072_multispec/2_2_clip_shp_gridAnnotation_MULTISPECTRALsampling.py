## PHASE 1_Image Processing
#STEP 2
# Step 2_2: Split orthomosaic raster to 3072x3072 px --> 10m x 10m (option b)
######### to get raster multispectral with annotated grid
from concurrent.futures import ThreadPoolExecutor
import os
from osgeo import gdal, ogr

def clip_raster_by_annotated_grid(input_raster_path, grid_shp_path, output_raster_folder, grid_id_field, annotated_field, annotated_value):
    # Ensure the output folder exists
    if not os.path.exists(output_raster_folder):
        os.makedirs(output_raster_folder)

    # Open the grid shapefile and raster
    grid_ds = ogr.Open(grid_shp_path)
    raster_ds = gdal.Open(input_raster_path)

    if not grid_ds or not raster_ds:
        raise ValueError("Could not open the files.")

    grid_layer = grid_ds.GetLayer()

    # Apply attribute filter to select grid features with specified 'annotated' value
    grid_layer.SetAttributeFilter(f"{annotated_field} = {annotated_value}")

    # Loop through each filtered grid feature
    for grid_feature in grid_layer:
        # Get the ID of the current grid feature
        grid_id = grid_feature.GetField(grid_id_field)

        # Prepare the output raster path
        output_raster_path = os.path.join(output_raster_folder, f"tiles_multispectral_{grid_id}.tif")

        # Clip raster
        clip_raster_feature(input_raster_path, grid_feature, output_raster_path)

    print("Raster clipping completed.")

def clip_raster_feature(input_raster_path, grid_feature, output_path):
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise RuntimeError("Unable to open input raster.")

    # Fetch the geometry of the grid feature and its envelope
    geom = grid_feature.GetGeometryRef()
    minX, maxX, minY, maxY = geom.GetEnvelope()

    # Define translation options for clipping
    translate_options = gdal.TranslateOptions(format="GTiff", outputType=gdal.GDT_Float32, projWin=[minX, maxY, maxX, minY])

    # Perform the clip and save the output
    gdal.Translate(output_path, src_ds, options=translate_options)




#site1_1 - supersite - DD0001 [MEDIUM] DONE
# input_raster_path = '/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/orthomosaic/20220519_SASMDD001_dual_ortho_01_bilinear.tif'
# grid_shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/griding/grid_annotation_3072.shp'
# output_raster_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site1_supersite_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral'

#site1_2 - similar to supersite - DD0010_18 [MEDIUM]
# input_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_DD0010_18/orthomosaic/20220518_SASMDD0010_18_dual_ortho_01_bilinear.tif'
# grid_shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/masks/tiles_3072/raw/pv_class/annotations/gridding/grid_annotation_3072.shp'
# output_raster_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_3072/raw/tiles_multispectral'

#site2_1 -  Vegetation - DD0011 [LOW] 
# input_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site4_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_01_bilinear.tif'
# grid_shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/masks/tiles_3072/raw/pv_class/annotations/gridding/grid_annotation_3072.shp'
# output_raster_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site4_DD0011/inputs/predictors/raw/tiles_multispectral'

#site3_1 - with water - DD0012 [DENSE] 
input_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_DD0012/orthomosaic/20220517_SASMDD0012_dual_ortho_01_bilinear.tif'
grid_shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/masks/tiles_3072/raw/pv_class/gridding/grid_annotation_3072.shp'
output_raster_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site3_DD0012/inputs/predictors/tiles_3072/raw/tiles_multispectral'

#site3_2 -similar to site 12 - DD0008 [DENSE]
# input_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site5_DD0008/orthomosaic/20220516_SASMDD0008_dual_ortho_01_bilinear.tif'
# grid_shapefile_path = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/masks/tiles_3072/raw/pv_class/annotations/gridding/grid_annotation_3072.shp'
# output_raster_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site5_DD0008/inputs/predictors/raw/tiles_multispectral'


grid_id_field = "id"  # Replace with your grid ID field name
annotated_field = "annotated"  # The name of the field used for filtering
annotated_value = 1  # The value in the 'annotated' field used for filtering


clip_raster_by_annotated_grid(input_raster_path, grid_shapefile_path, output_raster_folder, grid_id_field, annotated_field, annotated_value)