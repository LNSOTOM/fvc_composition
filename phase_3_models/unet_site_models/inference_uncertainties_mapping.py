
#%%
#### Uncertainty map with Monte Carlo dropout
# Release Cache
import gc

def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()
    
#%%
def check_memory_status():
    """Check memory status and perform garbage collection."""
    import gc
    import torch
    import psutil
    
    # Get initial memory stats
    if torch.cuda.is_available():
        initial_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
    initial_ram = psutil.Process().memory_info().rss / (1024 ** 3)
    
    # Run garbage collection and get result
    collected = gc.collect()
    
    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_diff = initial_gpu - final_gpu
        print(f"GPU Memory: Before {initial_gpu:.2f}GB, After {final_gpu:.2f}GB, Freed {gpu_diff:.2f}GB")
    
    # Check RAM after collection
    final_ram = psutil.Process().memory_info().rss / (1024 ** 3)
    ram_diff = initial_ram - final_ram
    
    print(f"RAM Memory: Before {initial_ram:.2f}GB, After {final_ram:.2f}GB, Freed {ram_diff:.2f}GB")
    print(f"Garbage collector removed {collected} objects")
    
    return collected

print(check_memory_status())

#%%
import psutil
import gc

def report_ram():
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024 ** 3)
    print(f"RAM used: {used_gb:.2f} GB of {mem.total / (1024**3):.2f} GB")

# Usage
report_ram()
gc.collect()
report_ram()



#%% Imports
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

    img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif' #low  (30, '33')'
    model_ckpt = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/low/original/block_2_epoch_96.pth'
    save_dir = 'wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_96'  # Directory to save the predictions low
    
    # img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/stacked/tiles_multispectral.101.tif'
    # save_dir = 'wombat_predictions_stitch_dense_1024_120ep_raw_bestmodel_105_tile101'
    # # model_ckpt = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/dense/original/block_3_epoch_105.pth'

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

# === Class label definitions ===
class_labels = {'BE': 0, 'NPV': 1, 'PV': 2, 'SI': 3, 'WI': 4}
inv_class_labels = {v: k for k, v in class_labels.items()}

# === File paths ===
unc_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_96/stitched_uncertainty.tif'
gt_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.33.tif'

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

# Choose a good sequential colormap for uncertainty
cmap_raw = plt.cm.coolwarm
# cmap_raw = plt.cm.inferno  # Or 'magma', 'viridis'

# Clip to reduce outlier influence
vmin_raw = np.nanpercentile(uncertainty_map, 1)
vmax_raw = np.nanpercentile(uncertainty_map, 99)

im = plt.imshow(uncertainty_map, cmap=cmap_raw, vmin=vmin_raw, vmax=vmax_raw)
plt.colorbar(im, label='Pixel-wise Model Uncertainty (MC Dropout Std. Dev)')
plt.title("Raw Epistemic Uncertainty Map (Model-Level)")
plt.axis('off')
plt.tight_layout()

# Save
raw_unc_png = 'raw_model_uncertainty_map.png'
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
vmin_mean = np.nanpercentile(mean_unc_map, 1)
vmax_mean = np.nanpercentile(mean_unc_map, 99)
im = plt.imshow(mean_unc_map, cmap=cmap_mean, vmin=vmin_mean, vmax=vmax_mean)
plt.colorbar(im, label='Mean Uncertainty (per class)')
plt.title("Per-Class Mean Uncertainty Map")
plt.axis('off')
plt.tight_layout()
mean_png = 'mean_uncertainty_per_class_map_styled.png'
plt.savefig(mean_png, dpi=300)
plt.show()
print(f"✅ Mean map PNG saved: {mean_png}")

# === Plot styled std dev map ===
plt.figure(figsize=(10, 8))
cmap_std = plt.cm.coolwarm
# cmap_std = plt.cm.BuPu  # Or use 'Blues'
vmin_std = np.nanpercentile(std_unc_map, 1)
vmax_std = np.nanpercentile(std_unc_map, 99)
im = plt.imshow(std_unc_map, cmap=cmap_std, vmin=vmin_std, vmax=vmax_std)
plt.colorbar(im, label='Uncertainty Std. Dev (per class)')
plt.title("Per-Class Standard Deviation of Uncertainty")
plt.axis('off')
plt.tight_layout()
std_png = 'std_uncertainty_per_class_map_styled.png'
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
plt.savefig("uncertainty_distribution_histogram.png", dpi=300)
plt.show()


# %%
