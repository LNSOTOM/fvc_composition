

#%%
#### Uncertainty map with Monte Carlo dropout
#Imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from torch.utils.data import DataLoader
from rasterio.windows import Window
import logging

from model.unet_module import UNetModule
from metrics.evaluation_bestmodel import ModelEvaluator
from dataset.calperum_dataset import CalperumDataset
from dataset.data_loaders_fold_blockcross_subsampling import get_dataset_splits
import config_param

# Set up logging
logging.basicConfig(level=logging.INFO)


"""
    Enables dropout layers during test-time inference to perform Monte Carlo sampling.
"""
# === Monte Carlo Dropout Utilities ===
def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


"""
    Performs Monte Carlo Dropout by running multiple stochastic forward passes through the model.

    Args:
        model (torch.nn.Module): Trained model with dropout layers.
        tile_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].
        num_passes (int): Number of stochastic forward passes.

    Returns:
        mean (torch.Tensor): Mean prediction across passes (shape: [B, C, H, W]).
        std (torch.Tensor): Standard deviation across passes (epistemic uncertainty).
"""
def mc_dropout_prediction(model, tile_tensor, num_passes=50):
    model.eval()
    enable_mc_dropout(model)
    preds = []

    with torch.no_grad():
        for _ in range(num_passes):
            output = model(tile_tensor)
            preds.append(output.unsqueeze(0))  # Shape: [1, B, C, H, W]

    preds = torch.cat(preds, dim=0)           # [T, B, C, H, W]
    mean = preds.mean(dim=0)                  # [B, C, H, W]
    std = preds.std(dim=0)                    # [B, C, H, W]
    return mean, std

def extract_large_area_tiles(img_path, tile_size=256):
    tiles, profiles, offsets = [], [], []

    with rasterio.open(img_path) as src:
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                transform = src.window_transform(window)
                profile = src.profile.copy()
                profile.update({
                    'width': min(tile_size, src.width - i),
                    'height': min(tile_size, src.height - j),
                    'transform': transform
                })
                tile = src.read(window=window)
                tiles.append(tile)
                profiles.append(profile)
                offsets.append((i, j))

    return tiles, profiles, offsets

def save_geotiff(image_array, profile, output_path):
    profile.update(dtype=rasterio.float32, count=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image_array, 1)

def run_inference_on_tiles(model, tiles, profiles, offsets, save_dir, device='cpu', num_passes=50):
    os.makedirs(save_dir, exist_ok=True)
    predictions = []

    for i, (tile, profile, (x_offset, y_offset)) in enumerate(zip(tiles, profiles, offsets)):
        tile_tensor = torch.tensor(tile).unsqueeze(0).float().to(device)  # [1, C, H, W]
        mean_pred, std_pred = mc_dropout_prediction(model, tile_tensor, num_passes)
        class_pred = torch.argmax(mean_pred, dim=1).cpu().numpy()[0]     # [H, W]
        uncertainty_map = torch.mean(std_pred, dim=1).cpu().numpy()[0]   # [H, W]

        save_geotiff(class_pred, profile, os.path.join(save_dir, f"tile_{i}_prediction.tif"))
        save_geotiff(uncertainty_map, profile, os.path.join(save_dir, f"tile_{i}_uncertainty.tif"))
        predictions.append((class_pred, uncertainty_map, profile, x_offset, y_offset))

    return predictions

def stitch_tiles(predictions, full_shape):
    pred_full = np.zeros(full_shape, dtype=np.float32)
    unc_full = np.zeros(full_shape, dtype=np.float32)

    for pred, unc, _, x_offset, y_offset in predictions:
        h, w = pred.shape
        pred_full[y_offset:y_offset+h, x_offset:x_offset+w] = pred
        unc_full[y_offset:y_offset+h, x_offset:x_offset+w] = unc

    return pred_full, unc_full

def run_large_area_inference(model, img_path, save_dir, tile_size=256, device='cpu', num_passes=50):
    tiles, profiles, offsets = extract_large_area_tiles(img_path, tile_size)
    predictions = run_inference_on_tiles(model, tiles, profiles, offsets, save_dir, device, num_passes)

    with rasterio.open(img_path) as src:
        full_shape = (src.height, src.width)

    pred_full, unc_full = stitch_tiles(predictions, full_shape)

    profile = profiles[0]
    profile.update({
        'height': full_shape[0],
        'width': full_shape[1],
        'transform': rasterio.transform.from_origin(
            profiles[0]['transform'].c, profiles[0]['transform'].f,
            profiles[0]['transform'].a, -profiles[0]['transform'].e)
    })

    save_geotiff(pred_full, profile, os.path.join(save_dir, 'stitched_prediction.tif'))
    save_geotiff(unc_full, profile, os.path.join(save_dir, 'stitched_uncertainty.tif'))

    logging.info("Saved stitched prediction and uncertainty map.")


#%%
def main():

    # img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif' #low  (30, '33')'
    # model_ckpt = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/low/original/block_2_epoch_96.pth'
    # model_ckpt = 'wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_96'  # Directory to save the predictions low
    
    img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.22.tif'
    save_dir = 'wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22'
    model_ckpt = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth'

    
    # img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/stacked/tiles_multispectral.118.tif'
    # save_dir = 'wombat_predictions_stitch_dense_1024_120ep_raw_bestmodel_42_tile118'
    # model_ckpt = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense_original_with_water/block_2_epoch_42.pth'

    model = UNetModule().to(config_param.DEVICE)
    model.load_state_dict(torch.load(model_ckpt, map_location=config_param.DEVICE))
    model.eval()

    logging.info("Running large area inference with MC Dropout...")
    run_large_area_inference(model, img_path, save_dir, tile_size=256, device=config_param.DEVICE, num_passes=50)

if __name__ == '__main__':
    main()

# %%
def plot_stitched_prediction(pred_map, unc_map=None):
    """
    Plots the stitched full-area prediction map and optionally the uncertainty map.
    """
    fig, axs = plt.subplots(1, 2 if unc_map is not None else 1, figsize=(14, 6))

    axs = np.atleast_1d(axs)
    axs[0].imshow(pred_map, cmap='tab20')
    axs[0].set_title("Stitched Prediction Map")
    axs[0].axis('off')

    if unc_map is not None:
        axs[1].imshow(unc_map, cmap='magma')
        axs[1].set_title("Stitched Uncertainty Map")
        axs[1].axis('off')

    plt.tight_layout()
    plt.show()


#%%
####  Uncertainty map with Monte Carlo dropout - stats
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# === Class label definitions ===
class_labels = {'BE': 0, 'NPV': 1, 'PV': 2, 'SI': 3, 'WI': 4}
inv_class_labels = {v: k for k, v in class_labels.items()}

# === File paths ===
# #low
# unc_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_96/stitched_uncertainty.tif'
# gt_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.33.tif'

# #medium
# unc_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22/stitched_uncertainty.tif'
# gt_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.22.tif'

#dense
unc_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/wombat_predictions_stitch_dense_1024_120ep_raw_bestmodel_42_tile118/stitched_uncertainty.tif'
gt_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.118.tif'


# === Load data ===
with rasterio.open(unc_path) as src:
    uncertainty_map = src.read(1).astype(np.float32)
    profile = src.profile

with rasterio.open(gt_path) as src:
    class_mask = src.read(1).astype(np.float32)

# === Prepare output maps ===
mean_unc_map = np.full_like(uncertainty_map, np.nan, dtype=np.float32)
std_unc_map = np.full_like(uncertainty_map, np.nan, dtype=np.float32)

print("Per-class statistics (excluding NaN/masked values):")
for class_val in sorted(class_labels.values()):
    class_name = inv_class_labels[class_val]
    
    # Only include pixels of this class and not masked (-1)
    mask = (class_mask == class_val) & (class_mask != -1)
    
    if np.any(mask):
        mean_val = np.mean(uncertainty_map[mask])
        std_val = np.std(uncertainty_map[mask])
        mean_unc_map[mask] = mean_val
        std_unc_map[mask] = std_val
        print(f"  {class_name} (class {class_val}): Mean = {mean_val:.4f}, Std = {std_val:.4f}")
    else:
        print(f"  {class_name} (class {class_val}): Not present in valid mask.")

# === Save mean and std maps ===
mean_tif = 'mean_uncertainty_per_class_map.tif'
std_tif = 'std_uncertainty_per_class_map.tif'
profile.update(dtype=rasterio.float32, count=1, compress='lzw')

with rasterio.open(mean_tif, 'w', **profile) as dst:
    dst.write(mean_unc_map, 1)
print(f"✅ Mean uncertainty GeoTIFF saved: {mean_tif}")

with rasterio.open(std_tif, 'w', **profile) as dst:
    dst.write(std_unc_map, 1)
print(f"✅ Std. dev. uncertainty GeoTIFF saved: {std_tif}")


# === Plot raw model uncertainty map ===
plt.figure(figsize=(10, 8))

# Choose a good sequential colormap
cmap_raw = plt.cm.coolwarm  # Alternatives: 'inferno', 'magma', 'viridis'
# cmap_raw = plt.cm.coolwarm_r  # ✅ Reverse colormap

# Clip to reduce outlier influence
vmin_raw = np.nanpercentile(uncertainty_map, 1)
vmax_raw = np.nanpercentile(uncertainty_map, 99)
vmid_raw = (vmin_raw + vmax_raw) / 2  # ✅ Midpoint

# Plot the uncertainty map
im = plt.imshow(uncertainty_map, cmap=cmap_raw, vmin=vmin_raw, vmax=vmax_raw)
plt.axis('off')
plt.tight_layout()

# Add colorbar with min, mid, max
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=20)
cbar.set_ticks([vmin_raw, vmid_raw, vmax_raw])
cbar.set_ticklabels([
    f'{vmin_raw:.3f}', 
    f'{vmid_raw:.3f}', 
    f'{vmax_raw:.3f}'
])

# Save the figure
raw_unc_png = 'raw_model_uncertainty_map_dense.png'
plt.savefig(raw_unc_png, dpi=300)
plt.show()

print(f"✅ Raw model uncertainty map saved: {raw_unc_png}")

# === Compute statistics of raw model uncertainty map ===
# Exclude NaN/masked areas using class_mask
valid_mask = (class_mask != -1)

# Flatten and mask uncertainty values
unc_values = uncertainty_map[valid_mask].flatten()

# Compute statistics
mean_unc = np.mean(unc_values)
std_unc = np.std(unc_values)

print("\n📊 Model-Level Uncertainty Statistics:")
print(f"  Mean uncertainty (std dev across MC Dropout): {mean_unc:.4f}")
print(f"  Std. deviation of uncertainty values:         {std_unc:.4f}")
print(f"  Number of valid pixels:                      {len(unc_values)}")


# === Plot styled mean map ===
plt.figure(figsize=(10, 8))
cmap_mean = plt.cm.coolwarm
# cmap_mean = plt.cm.coolwarm_r  # ✅ Reverse colormap
vmin_mean = np.nanpercentile(mean_unc_map, 1)
vmax_mean = np.nanpercentile(mean_unc_map, 99)
vmed_mean = (vmin_mean + vmax_mean) / 2  # ✅ Midpoint between min and max

im = plt.imshow(mean_unc_map, cmap=cmap_mean, vmin=vmin_mean, vmax=vmax_mean)
# plt.colorbar(im, label='Mean Uncertainty (per class)')
# plt.title("Per-Class Mean Uncertainty Map")
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=20)  # ✅ Set fontsize for colorbar tick labels

# ✅ Show only min and max values
cbar.set_ticks([vmin_mean, vmax_mean])
cbar.set_ticklabels([f'{vmin_mean:.3f}', f'{vmax_mean:.3f}'])  # Format as needed
# ✅ Set ticks: min, mid, max
cbar.set_ticks([vmin_mean, vmed_mean, vmax_mean])
cbar.set_ticklabels([
    f'{vmin_mean:.3f}',
    f'{vmed_mean:.3f}',
    f'{vmax_mean:.3f}'
])
plt.axis('off')
plt.tight_layout()
mean_png = 'mean_uncertainty_per_class_map_dense.png'
plt.savefig(mean_png, dpi=300)
plt.show()
print(f"✅ Mean map PNG saved: {mean_png}")

# === Plot styled std dev map ===
plt.figure(figsize=(10, 8))

# ✅ Start from white and blend into ColorBrewer's Blues
blues = cm.get_cmap('Blues', 256)
white_to_blue = LinearSegmentedColormap.from_list('white_to_blue', 
    [(0, 'white')] + [(i/255, blues(i)) for i in range(1, 256)], N=256)

# ✅ Set range
vmin_std = 0
vmax_std = np.nanpercentile(std_unc_map, 99)
vmid_std = (vmin_std + vmax_std) / 2

# Plot image
im = plt.imshow(std_unc_map, cmap=white_to_blue, vmin=vmin_std, vmax=vmax_std)

# Add colorbar
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=20)

# ✅ Show only min, mid, max ticks
cbar.set_ticks([vmin_std, vmid_std, vmax_std])
cbar.set_ticklabels([
    f'{vmin_std:.3f}', 
    f'{vmid_std:.3f}', 
    f'{vmax_std:.3f}'
])

plt.axis('off')
plt.tight_layout()

std_png = 'std_uncertainty_per_class_map_dense.png'
plt.savefig(std_png, dpi=300)
plt.show()

print(f"✅ Std. dev. map PNG saved: {std_png}")



# plot distribution
plt.figure(figsize=(8, 5))
sns.histplot(unc_values, bins=50, kde=True, color='orange')
plt.xlabel("Uncertainty (MC Dropout Std. Dev)")
plt.ylabel("Pixel Count")
plt.title("Distribution of Model Uncertainty")
plt.tight_layout()

# plt.savefig("uncertainty_distribution_histogram_medium.png", dpi=300)
plt.savefig("uncertainty_distribution_histogram_dense.png", dpi=300)
plt.show()


# %%
