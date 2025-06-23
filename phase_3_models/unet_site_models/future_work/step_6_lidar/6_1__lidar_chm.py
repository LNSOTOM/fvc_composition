# Drone LiDAR Processing Workflow in Python
#%%
import os
import numpy as np
import laspy
import subprocess
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import json

# === INPUT/OUTPUT PATHS ===
in_las_dir = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/"
out_dir = "/media/laura/Laura 102/fvc_composition/lidar_outputs/"
os.makedirs(out_dir, exist_ok=True)

las_file = os.path.join(in_las_dir, "cloud1457eb47b07af181.las")
ground_las_path = os.path.join(out_dir, "01_csf_ground.las")
norm_las_path = os.path.join(out_dir, "02_normalised.las")
pipeline_json = os.path.join(out_dir, "03_csf_pipeline.json")
dtm_tif_path = os.path.join(out_dir, "04_dtm.tif")
chm_tif_path = os.path.join(out_dir, "05_chm.tif")
canopy_cover_path = os.path.join(out_dir, "06_canopy_cover.tif")
canopy_density_path = os.path.join(out_dir, "07_canopy_density.tif")

#%%
# === STEP 1: CSF Ground Classification ===
def run_csf(input_las, output_las, pipeline_json,
            resolution=0.2, iterations=500, threshold=0.1, smooth=False):
    pipeline = [
        {"type": "readers.las", "filename": input_las},
        {"type": "filters.csf", "resolution": resolution,
         "iterations": iterations, "threshold": threshold, "smooth": smooth},
        {"type": "writers.las", "filename": output_las}
    ]
    os.makedirs(os.path.dirname(output_las), exist_ok=True)
    with open(pipeline_json, 'w') as f:
        json.dump(pipeline, f, indent=4)
    subprocess.run(["pdal", "pipeline", pipeline_json], check=True)
    print(f"✅ CSF output saved: {output_las}")

# === STEP 2: Height Normalization ===
def normalize_z(input_las, output_las):
    las = laspy.read(input_las)
    ground_mask = las.classification == 2
    if not np.any(ground_mask):
        raise ValueError("No ground-classified points found")
    min_z = np.min(las.z[ground_mask])
    z_norm = las.z - min_z
    if 'Z_norm' not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='Z_norm', type=np.float32))
    las.Z_norm = z_norm.astype(np.float32)
    las.write(output_las)
    print(f"✅ Normalized Z written: {output_las}")

# === STEP 3: Generate DTM ===
def generate_dtm(input_path, output_tif, res=0.05):
    las = laspy.read(input_path)
    mask = las.classification == 2
    x, y, z = np.array(las.x[mask]), np.array(las.y[mask]), np.array(las.z[mask])
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        raise ValueError("Input arrays for DTM generation are empty.")
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    dtm = df.groupby(['row', 'col'])['z'].min().unstack(fill_value=0)
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=dtm.shape[0],
                       width=dtm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
        dst.write(dtm.values.astype(np.float32), 1)
    print(f"✅ DTM raster saved: {output_tif}")

# === STEP 4: Generate CHM ===
def generate_chm(input_path, output_tif, res=0.05):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.Z_norm)
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        raise ValueError("Input arrays for CHM generation are empty.")
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    chm = df.groupby(['row', 'col'])['z'].max().unstack(fill_value=0)
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=chm.shape[0],
                       width=chm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
        dst.write(chm.values.astype(np.float32), 1)
    print(f"✅ CHM raster saved: {output_tif}")

# === STEP 5: Canopy Metrics ===
def compute_canopy_metrics(input_path, cover_tif, density_tif, threshold=1.4, res=1.0):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    df = pd.DataFrame({
        'x': np.array(las.x),
        'y': np.array(las.y),
        'z': np.array(las.Z_norm),
        'rn': np.array(las.return_number)
    })
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    grouped = df.groupby(['row', 'col'])
    cover = grouped.apply(lambda g: (g[g['rn'] == 1]['z'] > threshold).sum() / max(len(g[g['rn'] == 1]), 1))
    density = grouped.apply(lambda g: (g['z'] > threshold).sum() / len(g))
    shape = (df['row'].max() + 1, df['col'].max() + 1)
    cover_arr = np.full(shape, np.nan)
    density_arr = np.full(shape, np.nan)
    for (r, c), val in cover.items():
        cover_arr[r, c] = val
    for (r, c), val in density.items():
        density_arr[r, c] = val
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    for arr, path in zip([cover_arr, density_arr], [cover_tif, density_tif]):
        with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype='float32', crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
            dst.write(np.nan_to_num(arr).astype('float32'), 1)
        print(f"✅ Saved: {path}")


#%%
# === RUN PIPELINE ===
run_csf(las_file, ground_las_path, pipeline_json)
normalize_z(ground_las_path, norm_las_path)
generate_dtm(ground_las_path, dtm_tif_path)
generate_chm(norm_las_path, chm_tif_path)
compute_canopy_metrics(norm_las_path, canopy_cover_path, canopy_density_path)


#%%
## test 2
# Drone LiDAR Processing Workflow in Python

import os
import numpy as np
import laspy
import subprocess
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import json
from scipy.spatial import cKDTree

# === INPUT/OUTPUT PATHS ===
in_las_dir = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/"
out_dir = "/media/laura/Laura 102/fvc_composition/lidar_outputs/new"
os.makedirs(out_dir, exist_ok=True)

las_file = os.path.join(in_las_dir, "cloud1457eb47b07af181.las")
ground_las_path = os.path.join(out_dir, "01_csf_ground.las")
norm_las_path = os.path.join(out_dir, "02_normalised.las")
pipeline_json = os.path.join(out_dir, "03_csf_pipeline.json")
dtm_tif_path = os.path.join(out_dir, "04_dtm.tif")
chm_tif_path = os.path.join(out_dir, "05_chm.tif")
canopy_cover_path = os.path.join(out_dir, "06_canopy_cover.tif")
canopy_density_path = os.path.join(out_dir, "07_canopy_density.tif")

#%%
# === STEP 1: CSF Ground Classification ===
def run_csf(input_las, output_las, pipeline_json,
            resolution=0.2, iterations=500, threshold=0.1, smooth=False):
    pipeline = [
        {"type": "readers.las", "filename": input_las},
        {"type": "filters.csf", "resolution": resolution,
         "iterations": iterations, "threshold": threshold, "smooth": smooth},
        {"type": "writers.las", "filename": output_las}
    ]
    os.makedirs(os.path.dirname(output_las), exist_ok=True)
    with open(pipeline_json, 'w') as f:
        json.dump(pipeline, f, indent=4)
    subprocess.run(["pdal", "pipeline", pipeline_json], check=True)
    print(f"✅ CSF output saved: {output_las}")

# === STEP 2: Height Normalization ===
def normalize_z(input_las, output_las):
    las = laspy.read(input_las)
    ground_mask = las.classification == 2
    if not np.any(ground_mask):
        raise ValueError("No ground-classified points found")
    min_z = np.min(las.z[ground_mask])
    z_norm = las.z - min_z
    if 'Z_norm' not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='Z_norm', type=np.float32))
    las.Z_norm = z_norm.astype(np.float32)
    las.write(output_las)
    print(f"✅ Normalized Z written: {output_las}")

# === STEP 3: Generate DTM ===
def generate_dtm(input_path, output_tif, res=0.01):
    las = laspy.read(input_path)
    mask = las.classification == 2
    x, y, z = np.array(las.x[mask]), np.array(las.y[mask]), np.array(las.z[mask])
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        raise ValueError("Input arrays for DTM generation are empty.")
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    dtm = df.groupby(['row', 'col'])['z'].min().unstack(fill_value=0)
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=dtm.shape[0],
                       width=dtm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
        dst.write(dtm.values.astype(np.float32), 1)
    print(f"✅ DTM raster saved: {output_tif}")

# === STEP 4: Generate CHM (p2r-style with radius search) ===
# exclude ground points (classification == 2), ensuring that only canopy or above-ground features contribute to the CHM
# This better matches the standard CHM = DSM - DTM approach.
def generate_chm(input_path, output_tif, res=0.01, radius=0.2):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    non_ground_mask = las.classification != 2
    x, y, z = np.array(las.x[non_ground_mask]), np.array(las.y[non_ground_mask]), np.array(las.Z_norm[non_ground_mask])
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_grid = np.arange(x_min, x_max, res)
    y_grid = np.arange(y_min, y_max, res)
    chm = np.zeros((len(y_grid), len(x_grid)), dtype=np.float32)

    for i, y0 in enumerate(y_grid):
        for j, x0 in enumerate(x_grid):
            center = np.array([x0 + res / 2, y0 + res / 2])
            idx = tree.query_ball_point(center, r=radius)
            if idx:
                chm[i, j] = np.max(z[idx])
            else:
                chm[i, j] = 0.0

    transform = from_origin(x_min, y_max, res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=chm.shape[0],
                       width=chm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform) as dst:
        dst.write(chm, 1)
    print(f"✅ CHM (p2r-style, non-ground) raster saved: {output_tif}")

# === STEP 5: Canopy Metrics ===
def compute_canopy_metrics(input_path, cover_tif, density_tif, threshold=1.4, res=1.0):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    df = pd.DataFrame({
        'x': np.array(las.x),
        'y': np.array(las.y),
        'z': np.array(las.Z_norm),
        'rn': np.array(las.return_number)
    })
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    grouped = df.groupby(['row', 'col'])
    cover = grouped.apply(lambda g: (g[g['rn'] == 1]['z'] > threshold).sum() / max(len(g[g['rn'] == 1]), 1))
    density = grouped.apply(lambda g: (g['z'] > threshold).sum() / len(g))
    shape = (df['row'].max() + 1, df['col'].max() + 1)
    cover_arr = np.full(shape, np.nan)
    density_arr = np.full(shape, np.nan)
    for (r, c), val in cover.items():
        cover_arr[r, c] = val
    for (r, c), val in density.items():
        density_arr[r, c] = val
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    for arr, path in zip([cover_arr, density_arr], [cover_tif, density_tif]):
        with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype='float32', crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
            dst.write(np.nan_to_num(arr).astype('float32'), 1)
        print(f"✅ Saved: {path}")


#%%
# === RUN PIPELINE ===
run_csf(las_file, ground_las_path, pipeline_json)
normalize_z(ground_las_path, norm_las_path)
generate_dtm(ground_las_path, dtm_tif_path)
generate_chm(norm_las_path, chm_tif_path)
compute_canopy_metrics(norm_las_path, canopy_cover_path, canopy_density_path)


# %%
### TEST 3
# Drone LiDAR Processing Workflow in Python

import os
import numpy as np
import laspy
import subprocess
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import json
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

# === INPUT/OUTPUT PATHS ===
in_las_dir = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/"
reference_data = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/orthomosaic/'
out_dir = "/media/laura/Laura 102/fvc_composition/lidar_outputs/v2"
os.makedirs(out_dir, exist_ok=True)

las_file = os.path.join(in_las_dir, "cloud1457eb47b07af181.las")
reference_raster = os.path.join(reference_data, "20220519_SASMDD0001_p1_ortho_01.tif")
clipped_las_path = os.path.join(out_dir, "00_clipped_to_raster.las")
ground_las_path = os.path.join(out_dir, "01_csf_ground.las")
norm_las_path = os.path.join(out_dir, "02_normalised.las")
pipeline_json = os.path.join(out_dir, "03_csf_pipeline.json")
dtm_tif_path = os.path.join(out_dir, "04_dtm.tif")
chm_tif_path = os.path.join(out_dir, "05_chm.tif")
canopy_cover_path = os.path.join(out_dir, "06_canopy_cover.tif")
canopy_density_path = os.path.join(out_dir, "07_canopy_density.tif")


#%%
# === Clip LAS to Raster Extent and match resolution ===
def clip_lidar_to_raster_extent(las_input_path, raster_path, las_output_path):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        res = src.res[0]  # Extract resolution from reference raster

    las = laspy.read(las_input_path)
    mask = (
        (las.x >= bounds.left) & (las.x <= bounds.right) &
        (las.y >= bounds.bottom) & (las.y <= bounds.top)
    )

    if not np.any(mask):
        raise ValueError("No LiDAR points fall within the raster extent.")

    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    for dim in las.point_format.dimension_names:
        setattr(new_las, dim, getattr(las, dim)[mask])

    new_las.write(las_output_path)
    print(f"✅ Clipped LAS written to: {las_output_path}")
    # This step ensures all subsequent rasters align spatially and in resolution with the reference raster.
    return res

# === STEP 1: CSF Ground Classification ===
def run_csf(input_las, output_las, pipeline_json,
            resolution=0.2, iterations=500, threshold=0.1, smooth=False):
    pipeline = [
        {"type": "readers.las", "filename": input_las},
        {"type": "filters.csf", "resolution": resolution,
         "iterations": iterations, "threshold": threshold, "smooth": smooth},
        {"type": "writers.las", "filename": output_las}
    ]
    os.makedirs(os.path.dirname(output_las), exist_ok=True)
    with open(pipeline_json, 'w') as f:
        json.dump(pipeline, f, indent=4)
    subprocess.run(["pdal", "pipeline", pipeline_json], check=True)
    print(f"✅ CSF output saved: {output_las}")

# === STEP 2: Height Normalization ===
def normalize_z(input_las, output_las):
    las = laspy.read(input_las)
    ground_mask = las.classification == 2
    if not np.any(ground_mask):
        raise ValueError("No ground-classified points found")
    min_z = np.min(las.z[ground_mask])
    z_norm = las.z - min_z
    if 'Z_norm' not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='Z_norm', type=np.float32))
    las.Z_norm = z_norm.astype(np.float32)
    las.write(output_las)
    print(f"✅ Normalized Z written: {output_las}")

# === STEP 3: Generate DTM ===
def generate_dtm(input_path, output_tif, res):
    """
    Generate a DTM using minimum Z values for each raster cell.
    Note: This aligns with the R `rasterize_terrain` function using `knnidw(k=10, p=2)`.
    """
    las = laspy.read(input_path)
    mask = las.classification == 2
    x, y, z = np.array(las.x[mask]), np.array(las.y[mask]), np.array(las.z[mask])
    if len(x) == 0:
        raise ValueError("No ground points found for DTM generation.")
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    dtm = df.groupby(['row', 'col'])['z'].min().unstack(fill_value=0)
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=dtm.shape[0],
                       width=dtm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
        dst.write(dtm.values.astype(np.float32), 1)
    print(f"✅ DTM raster saved: {output_tif}")

# === STEP 4: Generate CHM ===
def generate_chm(input_path, output_tif, res, radius=0.2):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    non_ground_mask = las.classification != 2
    x, y, z = las.x[non_ground_mask], las.y[non_ground_mask], las.Z_norm[non_ground_mask]
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)

    x_grid = np.arange(x.min(), x.max(), res)
    y_grid = np.arange(y.min(), y.max(), res)
    chm = np.zeros((len(y_grid), len(x_grid)), dtype=np.float32)

    for i, y0 in enumerate(y_grid):
        for j, x0 in enumerate(x_grid):
            center = np.array([x0 + res / 2, y0 + res / 2])
            idx = tree.query_ball_point(center, r=radius)
            if idx:
                chm[i, j] = np.max(z[idx])
            else:
                chm[i, j] = 0.0

    transform = from_origin(x.min(), y.max(), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=chm.shape[0],
                       width=chm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform) as dst:
        dst.write(chm, 1)
    print(f"✅ CHM raster saved: {output_tif}")

# === STEP 5: Canopy Metrics ===
def compute_canopy_metrics(input_path, cover_tif, density_tif, threshold=1.4, res=1.0):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    df = pd.DataFrame({
        'x': las.x,
        'y': las.y,
        'z': las.Z_norm,
        'rn': las.return_number
    })
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    grouped = df.groupby(['row', 'col'])
    cover = grouped.apply(lambda g: (g[g['rn'] == 1]['z'] > threshold).sum() / max(len(g[g['rn'] == 1]), 1))
    density = grouped.apply(lambda g: (g['z'] > threshold).sum() / len(g))
    shape = (df['row'].max() + 1, df['col'].max() + 1)
    cover_arr = np.full(shape, np.nan)
    density_arr = np.full(shape, np.nan)
    for (r, c), val in cover.items():
        cover_arr[r, c] = val
    for (r, c), val in density.items():
        density_arr[r, c] = val
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    for arr, path in zip([cover_arr, density_arr], [cover_tif, density_tif]):
        with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype='float32', crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
            dst.write(np.nan_to_num(arr).astype('float32'), 1)
        print(f"✅ Saved: {path}")


#%%
# === RUN PIPELINE ===
res_from_raster = clip_lidar_to_raster_extent(las_file, reference_raster, clipped_las_path)
run_csf(clipped_las_path, ground_las_path, pipeline_json)
normalize_z(ground_las_path, norm_las_path)
generate_dtm(ground_las_path, dtm_tif_path, res=res_from_raster)
generate_chm(norm_las_path, chm_tif_path, res=res_from_raster)
compute_canopy_metrics(norm_las_path, canopy_cover_path, canopy_density_path, res=res_from_raster)

# %%
### test 4
# Drone LiDAR Processing Workflow in Python (Aligned to Reference Raster Resolution)
# Drone LiDAR Processing Workflow in Python (with Parallel Processing)

import os
import numpy as np
import laspy
import subprocess
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import json
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor

# === INPUT/OUTPUT PATHS ===
in_las_dir = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/"
reference_data = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/orthomosaic/'
out_dir = "/media/laura/Laura 102/fvc_composition/lidar_outputs/v2"
os.makedirs(out_dir, exist_ok=True)

las_file = os.path.join(in_las_dir, "cloud1457eb47b07af181.las")
reference_raster = os.path.join(reference_data, "20220519_SASMDD0001_p1_ortho_01.tif")
ground_las_path = os.path.join(out_dir, "01_csf_ground.las")
norm_las_path = os.path.join(out_dir, "02_normalised.las")
pipeline_json = os.path.join(out_dir, "03_csf_pipeline.json")
dtm_tif_path = os.path.join(out_dir, "04_dtm.tif")
chm_tif_path = os.path.join(out_dir, "05_chm.tif")
canopy_cover_path = os.path.join(out_dir, "06_canopy_cover.tif")
canopy_density_path = os.path.join(out_dir, "07_canopy_density.tif")


#%%
# === Get resolution from reference raster ===
def get_reference_resolution(raster_path):
    with rasterio.open(raster_path) as src:
        return float(src.res[0])


# === STEP 1: CSF Ground Classification ===
def run_csf(input_las, output_las, pipeline_json,
            resolution=0.2, iterations=500, threshold=0.1, smooth=False):
    pipeline = [
        {"type": "readers.las", "filename": input_las},
        {"type": "filters.csf", "resolution": resolution,
         "iterations": iterations, "threshold": threshold, "smooth": smooth},
        {"type": "writers.las", "filename": output_las}
    ]
    os.makedirs(os.path.dirname(output_las), exist_ok=True)
    with open(pipeline_json, 'w') as f:
        json.dump(pipeline, f, indent=4)
    subprocess.run(["pdal", "pipeline", pipeline_json], check=True)
    print(f"✅ CSF output saved: {output_las}")

# === STEP 2: Height Normalization ===
def normalize_z(input_las, output_las):
    las = laspy.read(input_las)
    ground_mask = las.classification == 2
    if not np.any(ground_mask):
        raise ValueError("No ground-classified points found")
    min_z = np.min(las.z[ground_mask])
    z_norm = las.z - min_z
    if 'Z_norm' not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='Z_norm', type=np.float32))
    las.Z_norm = z_norm.astype(np.float32)
    las.write(output_las)
    print(f"✅ Normalized Z written: {output_las}")

# === STEP 3: Generate DTM ===
def generate_dtm(input_path, output_tif, res):
    las = laspy.read(input_path)
    mask = las.classification == 2
    x, y, z = np.array(las.x[mask]), np.array(las.y[mask]), np.array(las.z[mask])
    if len(x) == 0:
        raise ValueError("No ground points found for DTM generation.")
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    dtm = df.groupby(['row', 'col'])['z'].min().unstack(fill_value=0)
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=dtm.shape[0],
                       width=dtm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
        dst.write(dtm.values.astype(np.float32), 1)
    print(f"✅ DTM raster saved: {output_tif}")

# === STEP 4: Generate CHM ===
def generate_chm(input_path, output_tif, res, radius=0.2):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    non_ground_mask = las.classification != 2
    x, y, z = las.x[non_ground_mask], las.y[non_ground_mask], las.Z_norm[non_ground_mask]
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)

    x_grid = np.arange(x.min(), x.max(), res)
    y_grid = np.arange(y.min(), y.max(), res)
    chm = np.zeros((len(y_grid), len(x_grid)), dtype=np.float32)

    def process_cell(i, j):
        center = np.array([x_grid[j] + res / 2, y_grid[i] + res / 2])
        idx = tree.query_ball_point(center, r=radius)
        return np.max(z[idx]) if idx else 0.0

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda ij: process_cell(*ij),
            [(i, j) for i in range(len(y_grid)) for j in range(len(x_grid))]
        ))

    chm = np.array(results, dtype=np.float32).reshape((len(y_grid), len(x_grid)))
    transform = from_origin(x.min(), y.max(), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=chm.shape[0],
                       width=chm.shape[1], count=1, dtype='float32',
                       crs='EPSG:7854', transform=transform) as dst:
        dst.write(chm, 1)
    print(f"✅ CHM raster saved: {output_tif}")

# === STEP 5: Canopy Metrics ===
def compute_canopy_metrics(input_path, cover_tif, density_tif, threshold=1.4, res=1.0):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"❌ 'Z_norm' dimension missing in: {input_path}. Did you run normalize_z()?")
    df = pd.DataFrame({
        'x': las.x,
        'y': las.y,
        'z': las.Z_norm,
        'rn': las.return_number
    })
    df['col'] = ((df.x - df.x.min()) // res).astype(int)
    df['row'] = ((df.y.max() - df.y) // res).astype(int)
    grouped = df.groupby(['row', 'col'])
    cover = grouped.apply(lambda g: (g[g['rn'] == 1]['z'] > threshold).sum() / max(len(g[g['rn'] == 1]), 1))
    density = grouped.apply(lambda g: (g['z'] > threshold).sum() / len(g))
    shape = (df['row'].max() + 1, df['col'].max() + 1)
    cover_arr = np.full(shape, np.nan)
    density_arr = np.full(shape, np.nan)
    for (r, c), val in cover.items():
        cover_arr[r, c] = val
    for (r, c), val in density.items():
        density_arr[r, c] = val
    transform = from_origin(df.x.min(), df.y.max(), res, res)
    for arr, path in zip([cover_arr, density_arr], [cover_tif, density_tif]):
        with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype='float32', crs='EPSG:7854', transform=transform, nodata=0.0) as dst:
            dst.write(np.nan_to_num(arr).astype('float32'), 1)
        print(f"✅ Saved: {path}")


#%%
# === RUN PIPELINE ===
res_from_raster = get_reference_resolution(reference_raster)
run_csf(las_file, ground_las_path, pipeline_json)
normalize_z(ground_las_path, norm_las_path)
generate_dtm(ground_las_path, dtm_tif_path, res=res_from_raster)
generate_chm(norm_las_path, chm_tif_path, res=res_from_raster)
compute_canopy_metrics(norm_las_path, canopy_cover_path, canopy_density_path, res=res_from_raster)

# %%
###test 5
# Drone LiDAR Processing Workflow in Python (Aligned to Reference Raster Resolution)

import os
import numpy as np
import laspy
import subprocess
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import json
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import time
from scipy.interpolate import griddata

# === INPUT/OUTPUT PATHS ===
# in_las_dir = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/"
# reference_data = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/orthomosaic/'
# out_dir = "/media/laura/Laura 102/fvc_composition/lidar_outputs/v2"
# os.makedirs(out_dir, exist_ok=True)

# las_file = os.path.join(in_las_dir, "cloud1457eb47b07af181.las")
# reference_raster = os.path.join(reference_data, "20220519_SASMDD0001_p1_ortho_01.tif")
# ground_las_path = os.path.join(out_dir, "01_csf_ground.las")
# norm_las_path = os.path.join(out_dir, "02_normalised.las")
# pipeline_json = os.path.join(out_dir, "03_csf_pipeline.json")
# dtm_tif_path = os.path.join(out_dir, "04_dtm.tif")
# chm_tif_path = os.path.join(out_dir, "05_chm.tif")
# canopy_cover_path = os.path.join(out_dir, "06_canopy_cover.tif")
# canopy_density_path = os.path.join(out_dir, "07_canopy_density.tif")

in_las_dir = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/"
reference_data = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral/'
out_dir = "/media/laura/Laura 102/fvc_composition/lidar_outputs/v2"
os.makedirs(out_dir, exist_ok=True)

las_file = os.path.join(in_las_dir, "cloud1457eb47b07af181.las")
reference_raster = os.path.join(reference_data, "tiles_multispectral.82.tif")
clipped_las_path = os.path.join(out_dir, "00_clipped_to_reference.las")
ground_las_path = os.path.join(out_dir, "01_csf_ground.las")
norm_las_path = os.path.join(out_dir, "02_normalised.las")
pipeline_json = os.path.join(out_dir, "03_csf_pipeline.json")
dtm_tif_path = os.path.join(out_dir, "04_dtm.tif")
chm_tif_path = os.path.join(out_dir, "05_chm.tif")
canopy_cover_path = os.path.join(out_dir, "06_canopy_cover.tif")
canopy_density_path = os.path.join(out_dir, "07_canopy_density.tif")


#%%
# === Get spatial parameters from reference raster ===
# === Get spatial parameters from reference raster ===
def get_reference_raster_params(raster_path):
    with rasterio.open(raster_path) as src:
        return src.transform, src.width, src.height, src.crs, src.res[0]

# === Clip LAS to Reference Raster Extent ===
def clip_las_to_raster_extent(input_las_path, raster_path, output_las_path):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    las = laspy.read(input_las_path)
    mask = (
        (las.x >= bounds.left) & (las.x <= bounds.right) &
        (las.y >= bounds.bottom) & (las.y <= bounds.top)
    )
    if not np.any(mask):
        raise ValueError("No LiDAR points within raster extent.")
    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    for dim in las.point_format.dimension_names:
        setattr(new_las, dim, getattr(las, dim)[mask])
    new_las.write(output_las_path)
    print(f"✅ LAS clipped to raster extent and saved: {output_las_path}")

# === STEP 1: CSF Ground Classification ===
def run_csf(input_las, csf_output_las, pipeline_json):
    pdal_pipeline = [
        {
            "type": "readers.las",
            "filename": input_las
        },
        {
            "type": "filters.csf",
            "resolution": 0.2,
            "iterations": 500,
            "threshold": 0.1,
            "smooth": False
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {
            "type": "writers.las",
            "filename": csf_output_las
        }
    ]

    with open(pipeline_json, "w") as f:
        json.dump(pdal_pipeline, f, indent=4)
    print(f"Pipeline JSON written to: {pipeline_json}")

    subprocess.run(["pdal", "pipeline", pipeline_json], check=True)
    print(f"Ground-classified LAS written to: {csf_output_las}")

# === STEP 2: Height Normalization - Option 1: Simple Min-Z subtraction ===
# def normalize_z(input_las, output_las):
#     las = laspy.read(input_las)
#     classification = las.classification
#     points = np.vstack([las.x, las.y, las.z]).T

#     ground_mask = classification == 2
#     ground_points = points[ground_mask]

#     if len(ground_points) == 0:
#         raise ValueError("❌ No ground points found in CSF output!")

#     min_ground_z = np.min(ground_points[:, 2])
#     z_normalized = las.z - min_ground_z

#     if "Z_norm" not in las.point_format.dimension_names:
#         las.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))

#     las.Z_norm = z_normalized.astype(np.float32)
#     las.write(output_las)

#     print(f"✅ Normalized LAS with Z_norm saved to: {output_las}")
    
# === STEP 2: Height Normalization (IDW) ===
def normalize_z(input_las, output_las):
    las = laspy.read(input_las)
    x, y, z = las.x, las.y, las.z
    classification = las.classification
    ground_mask = classification == 2
    ground_coords = np.column_stack((x[ground_mask], y[ground_mask]))
    ground_z = z[ground_mask]
    if len(ground_z) == 0:
        raise ValueError("❌ No ground points found in CSF output!")
    all_coords = np.column_stack((x, y))
    nn = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(ground_coords)
    dist, idx = nn.kneighbors(all_coords)
    weights = 1 / (dist ** 2 + 1e-6)
    interpolated_ground_z = np.sum(ground_z[idx] * weights, axis=1) / np.sum(weights, axis=1)
    z_normalized = z - interpolated_ground_z
    if "Z_norm" not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las.Z_norm = z_normalized.astype(np.float32)
    las.write(output_las)
    print(f"✅ IDW-normalized LAS with Z_norm saved to: {output_las}")
    
# === STEP 3: Generate DTM using IDW from ground points ===
def generate_dtm(input_path, output_tif, transform, width, height, crs, res):
    las = laspy.read(input_path)
    mask = las.classification == 2
    x, y, z = np.array(las.x[mask]), np.array(las.y[mask]), np.array(las.z[mask])
    if len(x) == 0:
        raise ValueError("No ground points found for DTM generation.")
    coords = np.vstack((x, y)).T
    tree = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', p=2).fit(coords)
    data = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            x0, y0 = rasterio.transform.xy(transform, row, col, offset='center')
            dist, idx = tree.kneighbors([[x0, y0]])
            if np.sum(dist) > 0:
                weights = 1 / (dist[0] ** 2)
                data[row, col] = np.sum(z[idx[0]] * weights) / np.sum(weights)
            else:
                data[row, col] = 0.0
    with rasterio.open(output_tif, 'w', driver='GTiff', height=height, width=width,
                       count=1, dtype='float32', crs=crs, transform=transform, nodata=0.0) as dst:
        dst.write(data, 1)
    print(f"✅ DTM raster saved: {output_tif}")
    
# def generate_dtm(input_path, output_tif, transform, width, height, crs, res):
#     las = laspy.read(input_path)
#     mask = las.classification == 2
#     x, y, z = np.array(las.x[mask]), np.array(las.y[mask]), np.array(las.z[mask])
#     if len(x) == 0:
#         raise ValueError("No ground points found for DTM generation.")
#     df = pd.DataFrame({'x': x, 'y': y, 'z': z})
#     df['col'] = ((df.x - transform.c) // res).astype(int)
#     df['row'] = ((transform.f - df.y) // res).astype(int)
#     df = df[(df['col'] >= 0) & (df['col'] < width) & (df['row'] >= 0) & (df['row'] < height)]
#     dtm = df.groupby(['row', 'col'])['z'].min().unstack(fill_value=0)
#     data = np.zeros((height, width), dtype=np.float32)
#     for r in dtm.index:
#         for c in dtm.columns:
#             data[r, c] = dtm.loc[r, c]
#     with rasterio.open(output_tif, 'w', driver='GTiff', height=height, width=width,
#                        count=1, dtype='float32', crs=crs, transform=transform, nodata=0.0) as dst:
#         dst.write(data, 1)
#     print(f"✅ DTM raster saved: {output_tif}")

# === Step 4: Generate CHM ===
# def generate_chm(input_path, output_tif, transform, width, height, crs, res, radius=0.2):
#     las = laspy.read(input_path)
#     if "Z_norm" not in las.point_format.dimension_names:
#         raise ValueError("Missing Z_norm. Run normalize_z first.")
#     mask = las.classification != 2
#     x, y, z = las.x[mask], las.y[mask], las.Z_norm[mask]
#     coords = np.column_stack((x, y))
#     tree = cKDTree(coords)
#     chm = np.zeros((height, width), dtype=np.float32)
#     for row in range(height):
#         for col in range(width):
#             x0, y0 = rasterio.transform.xy(transform, row, col, offset='center')
#             idx = tree.query_ball_point([x0, y0], r=radius)
#             chm[row, col] = np.max(z[idx]) if idx else 0.0
#     with rasterio.open(output_tif, 'w', driver='GTiff', height=height, width=width,
#                        count=1, dtype='float32', crs=crs, transform=transform) as dst:
#         dst.write(chm, 1)
#     print(f"✅ CHM raster saved: {output_tif}")

# === Step 4: Generate CHM using rasterized canopy height from p2r(0.2) ===
def generate_chm_p2r(input_path, output_tif, res=0.05, radius=0.2):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError("Missing Z_norm. Run normalize_z first.")
    x = las.x
    y = las.y
    z = las.Z_norm
    xi = np.arange(np.min(x), np.max(x), res)
    yi = np.arange(np.min(y), np.max(y), res)
    xi, yi = np.meshgrid(xi, yi)
    chm_interp = griddata((x, y), z, (xi, yi), method='nearest', fill_value=0.0)
    transform = from_origin(np.min(x), np.max(y), res, res)
    with rasterio.open(output_tif, 'w', driver='GTiff', height=chm_interp.shape[0], width=chm_interp.shape[1],
                       count=1, dtype='float32', crs='EPSG:7854', transform=transform) as dst:
        dst.write(chm_interp.astype(np.float32), 1)
    print(f"✅ CHM (p2r, r={radius}) raster saved: {output_tif}")

# === Step 5: Compute Canopy Metrics ===
def compute_canopy_metrics(input_path, cover_tif, density_tif, transform, width, height, crs, threshold=1.4, res=1.0):
    las = laspy.read(input_path)
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError("Missing Z_norm. Run normalize_z first.")
    df = pd.DataFrame({
        'x': las.x,
        'y': las.y,
        'z': las.Z_norm,
        'rn': las.return_number
    })
    col = ((df.x - transform.c) // res).astype(int)
    row = ((transform.f - df.y) // res).astype(int)
    df['col'], df['row'] = col, row
    grouped = df.groupby(['row', 'col'])
    cover = grouped.apply(lambda g: (g[g['rn'] == 1]['z'] > threshold).sum() / max(len(g[g['rn'] == 1]), 1))
    density = grouped.apply(lambda g: (g['z'] > threshold).sum() / len(g))

    cover_arr = np.full((height, width), np.nan)
    density_arr = np.full((height, width), np.nan)
    for (r, c), val in cover.items():
        if 0 <= r < height and 0 <= c < width:
            cover_arr[r, c] = val
    for (r, c), val in density.items():
        if 0 <= r < height and 0 <= c < width:
            density_arr[r, c] = val

    for arr, path in zip([cover_arr, density_arr], [cover_tif, density_tif]):
        with rasterio.open(path, 'w', driver='GTiff', height=height, width=width,
                           count=1, dtype='float32', crs=crs, transform=transform, nodata=0.0) as dst:
            dst.write(np.nan_to_num(arr).astype('float32'), 1)
        print(f"✅ Saved: {path}")


#%%
# === RUN PIPELINE ===
start = time.time()
transform, width, height, crs, res = get_reference_raster_params(reference_raster)
# clip_las_to_raster_extent(las_file, reference_raster, clipped_las_path)
# run_csf(clipped_las_path, ground_las_path, pipeline_json)
normalize_z(ground_las_path, norm_las_path)
generate_dtm(ground_las_path, dtm_tif_path, transform, width, height, crs, res)
generate_chm(norm_las_path, chm_tif_path, transform, width, height, crs, res)
compute_canopy_metrics(norm_las_path, canopy_cover_path, canopy_density_path, transform, width, height, crs, threshold=1.4, res=res)
print(f"✅ All processing completed in {time.time() - start:.2f} seconds")
# %%
