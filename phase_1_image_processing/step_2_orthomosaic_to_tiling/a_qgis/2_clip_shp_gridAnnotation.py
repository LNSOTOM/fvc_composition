
#%%
#### clip shp to grid annotation + obtain shp + raster output with Multispectral data + binary mask
from osgeo import ogr, osr, gdal
import os

def clip_vector_and_raster_by_annotated_grid(input_shapefile_path, input_raster_path, input_raster_path_2, grid_shapefile_path, output_vector_folder, output_raster_folder, output_raster_folder_2, grid_id_field, annotated_field, annotated_value):
   # Ensure the output folders exist
    for folder in [output_vector_folder, output_raster_folder, output_raster_folder_2]:
        if not os.path.exists(folder):
            os.makedirs(folder)


    # Open the input shapefile and grid shapefile
    input_ds = ogr.Open(input_shapefile_path)
    grid_ds = ogr.Open(grid_shapefile_path)
    raster_ds = gdal.Open(input_raster_path)
    raster_ds_2 = gdal.Open(input_raster_path_2)

    if not all([input_ds, grid_ds, raster_ds, raster_ds_2]):
        missing_files = [path for path, ds in [(input_shapefile_path, input_ds), (grid_shapefile_path, grid_ds), (input_raster_path, raster_ds), (input_raster_path_2, raster_ds_2)] if not ds]
        raise ValueError(f"Could not open the files: {', '.join(missing_files)}")
        

    input_layer = input_ds.GetLayer()
    grid_layer = grid_ds.GetLayer()

    # Fetch CRS from the input raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_ds.GetProjection())

    # Apply attribute filter to select grid features with 'annotated' = 1
    grid_layer.SetAttributeFilter(f"{annotated_field} = {annotated_value}")

    # Loop through each filtered grid feature
    for grid_feature in grid_layer:
        # Get the ID of the current grid feature
        grid_id = grid_feature.GetField(grid_id_field)

        # Prepare the output paths
        output_vector_path = os.path.join(output_vector_folder, f"tiles_multispectral_{grid_id}.shp")  #add the resolution such as "tiles_rgb_1024_{grid_id}.shp"
        output_raster_path = os.path.join(output_raster_folder, f"tiles_multispectral_{grid_id}.tif")
        output_raster_path_2 = os.path.join(output_raster_folder_2, f"mask_tiles_multispectral_{grid_id}.tif")


        # Clip vector
        clip_vector_feature(input_layer, grid_feature, output_vector_path, raster_srs)
        # Clip raster
        clip_raster_feature(input_raster_path, grid_feature, output_raster_path)      
        # New call to clip the second raster
        clip_raster_feature(input_raster_path_2, grid_feature, output_raster_path_2)
        
    print("Processing completed.")


def clip_vector_feature(input_layer, grid_feature, output_path, srs):
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Determine the geometry type of the input layer for creating a compatible output layer
    geom_type = input_layer.GetGeomType()
    
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    output_ds = driver.CreateDataSource(output_path)
    
    # Create an output layer that matches the geometry type of the input layer
    output_layer = output_ds.CreateLayer("clipped_layer", srs, geom_type=geom_type)
    output_layer.CreateFields(input_layer.schema)

    grid_geom = grid_feature.GetGeometryRef()
    input_layer.SetSpatialFilter(grid_geom)

    for in_feature in input_layer:
        in_geom = in_feature.GetGeometryRef()
        if in_geom is None:
            continue  # Skip features without geometry

        intersection_geom = in_geom.Intersection(grid_geom)
        
        # Check if the intersection result is None or if it results in an empty or invalid geometry
        if intersection_geom is None or intersection_geom.IsEmpty() or not intersection_geom.IsValid():
            continue
        
        # Create a new feature for the output layer with the intersection geometry
        out_feature = ogr.Feature(output_layer.GetLayerDefn())
        out_feature.SetGeometry(intersection_geom)
        for i in range(out_feature.GetFieldCount()):
            out_feature.SetField(i, in_feature.GetField(i))
        output_layer.CreateFeature(out_feature)
    
    input_layer.SetSpatialFilter(None)


# MULTISPECTRAL DATA
def clip_raster_feature(input_raster_path, grid_feature, output_path):
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise RuntimeError("Unable to open input raster.")

    # Fetch the geometry of the grid feature and its envelope
    geom = grid_feature.GetGeometryRef()
    minX, maxX, minY, maxY = geom.GetEnvelope()

    # Define translation options
    # -ot Float32 to specify output data type as float32
    # -projWin to specify the clipping bounds (minX, maxY, maxX, minY corresponds to the upper left and lower right corners)
    translate_options = gdal.TranslateOptions(format="GTiff", outputType=gdal.GDT_Float32, projWin=[minX, maxY, maxX, minY], outputSRS=src_ds.GetProjection())

    # Perform the clip and save the output
    gdal.Translate(output_path, src_ds, options=translate_options)

    
#  file paths 
if __name__ == "__main__":
    # low site
    # input_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.79.shp'
    # input_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_01_bilinear.tif'
    # input_raster_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.79.tif'
    # grid_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/pv_class/annotations/gridding/gridding_79/grid_annotation_256.shp'
  
    # output_vector_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_shp_10b/shp_10b_79'
    # output_raster_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_raster_10b/raster_10b_79'
    # output_raster_folder_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_256/raw/fvc_class/mask_fvc/mask_fvc_79'
    
    
    # medium site
    # input_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.205.shp'
    # input_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/orthomosaic/20220519_SASMDD001_dual_ortho_01_bilinear.tif'
    # input_raster_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.205.tif'
    # grid_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/pv_class/annotations/griding/gridding_205/grid_annotation_256.shp'
  
    # output_vector_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_shp_10b/shp_10b_205'
    # output_raster_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_raster_10b/raster_10b_205'
    # output_raster_folder_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_256/raw/fvc_class/mask_fvc/mask_fvc_205'
    
    
    # dense site
    input_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.128.shp'
    input_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/orthomosaic/20220517_SASMDD0012_dual_ortho_01_bilinear.tif'    
    input_raster_path_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.128.tif'   
    grid_shapefile_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/pv_class/gridding/gridding_128/grid_annotation_256.shp'
  
    output_vector_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_shp_10b/shp_10b_128'
    output_raster_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_raster_10b/raster_10b_128'
    output_raster_folder_2 = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_256/raw/fvc_class/mask_fvc/mask_fvc_128'
    



grid_id_field = "id"  # Replace "ID" with the actual field name of the grid ID in your shapefile
annotated_field = "annotated"  # The name of the field used for filtering in the grid shapefile
annotated_value = 1  # The value in the 'annotated' field used for filtering


clip_vector_and_raster_by_annotated_grid(input_shapefile_path, input_raster_path, input_raster_path_2, grid_shapefile_path, output_vector_folder, output_raster_folder, output_raster_folder_2, grid_id_field, annotated_field, annotated_value)

