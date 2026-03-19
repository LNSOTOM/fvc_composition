## PHASE 1_Image Processing
#STEP 5
# Step 5_2: Stack Bands (10b to 5b)
import os

import rasterio
from osgeo import gdal

## Multispectral imagery from MicaSense RedEdge-MX or Dual sensors 
# 'BAND NAMES', ['Blue-444', 'Blue', 'Green-531', 'Green', 'Red-650', 'Red', 'Red edge-705', 'Red edge', 'Red edge-740', 'NIR']
# 'WAVELENGTH', [444.0 , 475.0 , 531.0 , 560.0 , 650.0 , 668.0 , 705.0 , 717.0 , 740.0 , 842.0]
############### [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ['Blue', 'Green', 'Red', 'Red edge', 'NIR']
# [475.0, 560.0, 668.0, 717.0, 842.0]

####### VERSION IMPROVED TO ONLY STORE 5B AND NOT ALL 10B


def _pick_first(tags: dict, keys: list[str]) -> str | None:
    for k in keys:
        for kk in (k, k.lower(), k.upper()):
            v = tags.get(kk)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return None


def _format_nm(value: str) -> str | None:
    try:
        nm = float(str(value).strip())
    except Exception:
        return None
    if nm <= 0:
        return None
    if abs(nm - round(nm)) < 1e-6:
        return str(int(round(nm)))
    return f"{nm:g}"


def _band_description_from_metadata(
    *, src: rasterio.io.DatasetReader, gdal_ds, band_index: int
) -> str | None:
    # 1) rasterio-provided per-band description (often comes from GDAL)
    try:
        descs = list(src.descriptions or [])
        d = descs[band_index - 1] if band_index - 1 < len(descs) else None
        if d and str(d).strip():
            return str(d).strip()
    except Exception:
        pass

    # 2) band tags
    try:
        tags = src.tags(band_index) or {}
        description = _pick_first(tags, ["description", "DESCRIPTION", "band_description", "long_name"])
        if description:
            return description

        name = _pick_first(tags, ["band_name", "BAND_NAME", "bandname", "BANDNAME", "name", "NAME"])
        wavelength = _pick_first(tags, ["wavelength", "WAVELENGTH", "center_wavelength", "CENTRAL_WAVELENGTH"])
        nm = _format_nm(wavelength) if wavelength else None
        if name and nm:
            return f"Band {band_index:02d}: {name} [{nm} nm]"
        if name:
            return f"Band {band_index:02d}: {name}"
        if nm:
            return f"Band {band_index:02d}: [{nm} nm]"
    except Exception:
        pass

    # 3) GDAL band description/metadata
    try:
        band = gdal_ds.GetRasterBand(band_index)
        if band is not None:
            d = band.GetDescription()
            if d and str(d).strip():
                return str(d).strip()

            md = band.GetMetadata() or {}
            description = _pick_first(md, ["description", "DESCRIPTION", "band_description", "long_name"])
            if description:
                return description
            name = _pick_first(md, ["band_name", "BAND_NAME", "bandname", "BANDNAME", "name", "NAME"])
            wavelength = _pick_first(md, ["wavelength", "WAVELENGTH", "center_wavelength", "CENTRAL_WAVELENGTH"])
            nm = _format_nm(wavelength) if wavelength else None
            if name and nm:
                return f"Band {band_index:02d}: {name} [{nm} nm]"
            if name:
                return f"Band {band_index:02d}: {name}"
            if nm:
                return f"Band {band_index:02d}: [{nm} nm]"
    except Exception:
        pass

    return None


def _band_descriptions_for_selection(
    *, src: rasterio.io.DatasetReader, gdal_ds, selected_bands: list[int]
) -> list[str]:
    out: list[str] = []
    for b in selected_bands:
        d = _band_description_from_metadata(src=src, gdal_ds=gdal_ds, band_index=b)
        out.append(d if d else f"Band {b:02d}")
    return out

#sample data
# input_folder = '/home/laura/Documents/uas_data/Calperum/sampleData/site3_DD0012/inputs/predictors/tiles_multispectral'
# output_folder_5b = '/home/laura/Documents/uas_data/Calperum/sampleData/site3_DD0012/inputs/predictors/predictors_multispectral_5b/'

# Input and output folders
#site1 - supersite [MEDIUM]
##256x256
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_raster_10b/raster_10b_182/'
# output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_256/raw/predictors_5b/'

#site2 - similar to supersite [MEDIUM]
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_multispectral'
# output_folder_5b = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/predictors_multispectral_5b/'

##site3 - with water  [DENSE]
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_raster_10b/raster_10b_93/'
# output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_256/raw/predictors_5b/'

##site4 -  Vegetation [LOW] (potentially site DD0013)
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_raster_10b/raster_10b_25'
# output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_256/raw/predictors_5b/new/'


# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral'
# output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral/annotation_predictor_raster_5b/'

##site5 -  Vegetation [LOW] (site DD0013)
input_folder = '/media/laura/laura_usb/uas_data/DD0013/inputs/predictors/tiles_3072/raw/tiles_multispectral'
output_folder_5b = '/media/laura/laura_usb/uas_data/DD0013/inputs/predictors/tiles_3072/stacked/'


# Check for directory existence, if not, create
os.makedirs(output_folder_5b, exist_ok=True)
# os.makedirs(output_folder_5b, exist_ok=True)

# Bands to save for 10-band files
bands_to_save_10b = [2, 4, 6, 8, 10]

# List all files in the input folder
file_list = os.listdir(input_folder)

for input_filename in file_list:
    try:
        input_path = os.path.join(input_folder, input_filename)

        with rasterio.open(input_path) as src:
            # Error check if raster file is not in expected format
            if src.count < len(bands_to_save_10b) or src.count < 5:
                print(f"Skipping {input_filename}: It does not meet band selection criteria.")
                continue
            
            # Storing selected and reduced bands using in-memory slicing
            profile = src.profile
            # Read the data for 10b output
            data_10b = [src.read(b) for b in bands_to_save_10b]
            gdal_ds = gdal.Open(input_path)
            out_descriptions = _band_descriptions_for_selection(
                src=src, gdal_ds=gdal_ds, selected_bands=bands_to_save_10b
            )
            # Use pre-read file for 5b operation to pick starting bands based on present count
            start_5b_bands = list(range(1, min(6, src.count + 1)))
            data_5b = [src.read(i) for i in start_5b_bands]

        # Creating 10-band stacked Raster
        profile.update(count=len(data_10b))
        output_path_10b = os.path.join(output_folder_5b, f"{input_filename}")
        with rasterio.open(output_path_10b, 'w', **profile) as dst:
            for idx, data in enumerate(data_10b, start=1):
                dst.write(data, idx)
            dst.descriptions = tuple(out_descriptions)
                

    except Exception as e:
        print(f"Error occurred with {input_filename}: {e}")

print("Raster band reduction and saving are complete.")


###################################  3072PX STACKED
#%%
####### VERSION IMPROVED TO ONLY STORE 5B AND NOT ALL 10B

#sample data
# input_folder = '/home/laura/Documents/uas_data/Calperum/sampleData/site3_DD0012/inputs/predictors/tiles_multispectral'
# output_folder_5b = '/home/laura/Documents/uas_data/Calperum/sampleData/site3_DD0012/inputs/predictors/predictors_multispectral_5b/'

# Input and output folders
#site1 - supersite [MEDIUM]
##3072x3072
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral'
# output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/'

#site2 - similar to supersite [MEDIUM]
# input_folder = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/tiles_multispectral'
# output_folder_5b = '/home/laura/Documents/uas_data/Calperum/randomSamplingData/site2_DD0010_18/inputs/predictors/predictors_multispectral_5b/'

##site3 - with water  [DENSE]
# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_256/raw/fvc_class/annotation_predictor_raster_10b/raster_10b_93/'
# output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_256/raw/predictors_5b/'

##site4 -  Vegetation [LOW] (potentially site DD0013)
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral'
output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/'

# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral'
# output_folder_5b = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral/annotation_predictor_raster_5b/'


# Check for directory existence, if not, create
os.makedirs(output_folder_5b, exist_ok=True)
# os.makedirs(output_folder_5b, exist_ok=True)

# Bands to save for 10-band files
bands_to_save_10b = [2, 4, 6, 8, 10]

# List all files in the input folder
file_list = os.listdir(input_folder)

for input_filename in file_list:
    try:
        input_path = os.path.join(input_folder, input_filename)

        with rasterio.open(input_path) as src:
            # Error check if raster file is not in expected format
            if src.count < len(bands_to_save_10b) or src.count < 5:
                print(f"Skipping {input_filename}: It does not meet band selection criteria.")
                continue
            
            # Storing selected and reduced bands using in-memory slicing
            profile = src.profile
            # Read the data for 10b output
            data_10b = [src.read(b) for b in bands_to_save_10b]
            gdal_ds = gdal.Open(input_path)
            out_descriptions = _band_descriptions_for_selection(
                src=src, gdal_ds=gdal_ds, selected_bands=bands_to_save_10b
            )
            # Use pre-read file for 5b operation to pick starting bands based on present count
            start_5b_bands = list(range(1, min(6, src.count + 1)))
            data_5b = [src.read(i) for i in start_5b_bands]

        # Creating 10-band stacked Raster
        profile.update(count=len(data_10b))
        output_path_10b = os.path.join(output_folder_5b, f"{input_filename}")
        with rasterio.open(output_path_10b, 'w', **profile) as dst:
            for idx, data in enumerate(data_10b, start=1):
                dst.write(data, idx)
            dst.descriptions = tuple(out_descriptions)
                

    except Exception as e:
        print(f"Error occurred with {input_filename}: {e}")

print("Raster band reduction and saving are complete.")

# %%
