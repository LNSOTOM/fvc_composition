
#%%

# Define paths - good
import os
import numpy as np
import rasterio

# input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample'
# output_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample_nowater'

input_folder = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
output_folder = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc_nowater'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with rasterio.open(input_path) as src:
            mask = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            # Mask class 4 as NaN
            mask[mask == 4] = np.nan

            # Update profile to float32 and assign explicit nodata
            profile.update(
                dtype=rasterio.float32,
                nodata=np.nan
            )

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        print(f"Masked class 4 as NaN in: {filename}")


# %%
#%%
import os
import numpy as np
import rasterio

# Define paths
input_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample'
output_folder = '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample_nowater'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with rasterio.open(input_path) as src:
            mask = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            # Mask class 4 as NaN
            mask[mask == 4] = np.nan

            # Update profile to float32 (do not set nodata=np.nan)
            profile.update(dtype=rasterio.float32)
            if 'nodata' in profile:
                del profile['nodata']

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        print(f"Masked class 4 as NaN in: {filename}")

# %%
