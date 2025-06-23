#%%
import os
import json
import subprocess

# === Input/Output paths ===
laz_input = "/media/laura/Laura 102/fvc_composition/sfm_p1/20220517_SASMDD0012_p1_point_cloud.laz"         # Replace with your actual .laz file
chm_output = "/media/laura/Laura 102/fvc_composition/sfm_p1/chm/20220517_SASMDD0012_chm.tif"              # Final output (GeoTIFF)

# Ensure the output directory exists
output_dir = os.path.dirname(chm_output)
if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir)

dem_output_dir = "/media/laura/Laura 102/fvc_composition/sfm_p1/dem"
if not os.path.exists(dem_output_dir):
    print(f"Creating DEM output directory: {dem_output_dir}")
    os.makedirs(dem_output_dir)

# === Define PDAL pipeline as a dictionary ===
pdal_pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": laz_input
        },
        {
            "type": "filters.csf",
            "cloth_resolution": 0.5,
            "rigidness": 3,
            "threshold": 0.5  # Corrected parameter
        },
        {
            "type": "filters.hag_delaunay"
        },
        {
            "type": "writers.gdal",
            "filename": "/media/laura/Laura 102/fvc_composition/sfm_p1/dem/20220517_SASMDD0012_dem.tif",
            "resolution": 0.1,
            "output_type": "min",
            "dimension": "Z",
            "data_type": "float32",
            "gdaldriver": "GTiff"
        },
        {
            "type": "filters.hag_delaunay"
        },
        {
            "type": "filters.range",
            "limits": "Classification![2:2],HeightAboveGround[0.5:]"
        },
        {
            "type": "writers.gdal",
            "filename": chm_output,
            "resolution": 0.1,
            "radius": 0.5,
            "output_type": "max",
            "dimension": "HeightAboveGround",
            "data_type": "float32",
            "gdaldriver": "GTiff"
        },
        {
            "type": "writers.gdal",
            "filename": "/media/laura/Laura 102/fvc_composition/sfm_p1/dsm/20220517_SASMDD0012_dsm.tif",
            "resolution": 0.1,
            "output_type": "max",
            "dimension": "Z",
            "data_type": "float32",
            "gdaldriver": "GTiff"
        }
    ]
}

# Save pipeline to JSON file
with open("chm_full_pipeline.json", "w") as f:
    json.dump(pdal_pipeline, f, indent=4)

# Run PDAL pipeline
print(" Running PDAL pipeline to generate CHM...")
try:
    subprocess.run(["pdal", "pipeline", "chm_full_pipeline.json"], check=True)
    print(f"✅ CHM saved to: {chm_output}")
except subprocess.CalledProcessError as e:
    print(f"❌ PDAL pipeline failed with error: {e}")
# %%
