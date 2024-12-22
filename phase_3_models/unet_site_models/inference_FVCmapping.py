#%%
# Release Cache
import gc
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

#%%
########## 12.b enhancement' for list subdirectories
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.unet_module import UNetModule
from metrics.evaluation_bestmodel import ModelEvaluator
import config_param
from dataset.calperum_dataset import CalperumDataset
from dataset.data_loaders_fold_blockcross_subsampling import get_dataset_splits, block_cross_validation
from matplotlib.colors import ListedColormap
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def extract_large_area_tiles(img_path, tile_size=256):
    """Extract tiles from a large image and compute min/max for bands."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path {img_path} does not exist.")
    
    tiles = []
    profiles = []
    offsets = []

    with rasterio.open(img_path) as src:
        width = src.width
        height = src.height
        if width is None or height is None:
            raise ValueError("Failed to read image dimensions. Please check the input image file.")

        for j in range(0, height, tile_size):
            for i in range(0, width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                transform = src.window_transform(window)
                profile = src.profile
                profile.update({
                    'width': min(tile_size, src.width - i),
                    'height': min(tile_size, src.height - j),
                    'transform': transform
                })
                
                tile = src.read(window=window)
                if tile is None:
                    logging.warning(f"Failed to read tile at position ({i}, {j}). Skipping this tile.")
                    continue
                
                tiles.append(tile)
                profiles.append(profile)
                offsets.append((i, j))
    
    if not tiles:
        raise ValueError("No tiles were extracted. Please check the input image and parameters.")
    
    return tiles, profiles, offsets

def run_inference_on_tiles(model, tiles, profiles, offsets, save_dir, device='cpu'):
    if not tiles or not profiles or not offsets:
        raise ValueError("Invalid input for tiles, profiles, or offsets. Ensure these are correctly initialized.")
    
    os.makedirs(save_dir, exist_ok=True)
    predictions = []
    
    for i, (tile, profile, (x_offset, y_offset)) in enumerate(zip(tiles, profiles, offsets)):
        if tile is None or profile is None:
            logging.warning(f"Skipping tile {i} due to missing data.")
            continue
        
        tile_tensor = torch.tensor(tile).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            output = model(tile_tensor)
        preds = torch.argmax(output, dim=1).cpu().numpy()[0]  # Shape: (H, W)
        
        # Save the tile prediction as a GeoTIFF
        output_path = os.path.join(save_dir, f"tile_{i}_prediction.tif")
        save_geotiff(preds, profile, output_path)
        logging.info(f"Saved inference output for tile {i} to {output_path}")

        predictions.append((preds, profile, x_offset, y_offset))

    return predictions

def save_geotiff(image_array, profile, output_path):
    """ Save a single-band image array as a GeoTIFF with class labels. """
    if image_array is None or profile is None:
        raise ValueError("Invalid input for image_array or profile. Ensure these are correctly initialized.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Update the profile to match the single-band output
    profile.update(dtype=rasterio.float32, count=1)  # Single-band output

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image_array, 1)  # Write the array as band 1 (GeoTIFF bands are 1-based)

    logging.info(f"Single-band GeoTIFF saved to {output_path}")

def stitch_tiles(predictions, full_image_shape):
    """Stitch tiles into a full image based on their offsets."""
    if predictions is None or full_image_shape is None:
        raise ValueError("Invalid input for predictions or full_image_shape. Ensure these are correctly initialized.")
    
    full_image = np.zeros(full_image_shape, dtype=np.float32)
    
    for pred, profile, x_offset, y_offset in predictions:
        if pred is None:
            logging.warning(f"Skipping prediction at offset ({x_offset}, {y_offset}) due to missing data.")
            continue
        
        # Get the shape of the prediction
        h, w = pred.shape
        
        # Calculate the exact start and end positions in the full image
        start_row = y_offset
        end_row = start_row + h
        start_col = x_offset
        end_col = start_col + w

        # Ensure that the indices are within bounds
        if end_row <= full_image_shape[0] and end_col <= full_image_shape[1]:
            full_image[start_row:end_row, start_col:end_col] = pred
        else:
            logging.warning(f"Skipping tile due to out-of-bounds placement: {x_offset},{y_offset}")
    
    return full_image

def run_large_area_inference(model, img_path, save_dir, tile_size=256, device='cpu'):
    tiles, profiles, offsets = extract_large_area_tiles(img_path, tile_size=tile_size)
    predictions = run_inference_on_tiles(model, tiles, profiles, offsets, save_dir, device=device)
    
    # Full image dimensions
    with rasterio.open(img_path) as src:
        full_image_shape = (src.height, src.width)

    if not predictions:
        raise ValueError("No predictions were generated. Please check the inference process.")

    full_image = stitch_tiles(predictions, full_image_shape)

    if full_image is None:
        raise ValueError("Failed to stitch tiles into a full image. Please check the tile stitching process.")

    # Use the profile from the first tile, and adjust it for the full image
    full_profile = profiles[0]
    full_profile.update({
        'height': full_image_shape[0],
        'width': full_image_shape[1],
        'transform': rasterio.transform.from_origin(
            profiles[0]['transform'].c, profiles[0]['transform'].f,
            profiles[0]['transform'].a, -profiles[0]['transform'].e
        )
    })
    
    # stitched_output_path = os.path.join(save_dir, 'stitched_low_1024_120ep_raw_33.tif') #low
    # stitched_output_path = os.path.join(save_dir, 'stitched_medium_1024_120ep_raw_22.tif') #medium
    stitched_output_path = os.path.join(save_dir, 'stitched_dense_1024_120ep_raw_118.tif') #dense
    
    save_geotiff(full_image, full_profile, stitched_output_path)
    logging.info(f"Stitched image saved to {stitched_output_path}")
    
def main():
    try:
        # Define parameters
        # img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif' #low  (30, '33')
        # img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.22.tif' # medium ('22', 26)
        img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/stacked/tiles_multispectral.101.tif'  # dense (30, '118', 101)
     
        tile_size = 256  # Size of the tiles
        
        # save_dir = 'wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_96'  # Directory to save the predictions low
        # save_dir = 'wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_108'  # Directory to save the predictions low - better
        # save_dir = 'wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55'  # Directory to save the predictions medium
        # save_dir = 'wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_117'  # Directory to save the predictions medium - better
        # save_dir = 'wombat_predictions_stitch_dense_1024_120ep_raw_bestmodel_105'  # Directory to save the predictions dense
        save_dir = 'wombat_predictions_stitch_dense_1024_120ep_raw_bestmodel_105_tile101' 
        
        soil_threshold = 50.0
        removal_ratio = 0.9
        num_classes = config_param.OUT_CHANNELS
        
        # Initialize lists for images and masks
        all_images = []
        all_masks = []
        all_subsampled_indices = []

        # Check if subsampled data exists in all directories
        subsampled_data_exists = all(os.path.exists(dir) and os.listdir(dir) for dir in config_param.SUBSAMPLE_IMAGE_DIR) and \
                                 all(os.path.exists(dir) and os.listdir(dir) for dir in config_param.SUBSAMPLE_MASK_DIR)
        
        if subsampled_data_exists:
            logging.info(f"Subsampled image and mask files found in the specified directories, loading data...")
            
            # Load the subsampled images and masks from all directories
            for image_dir, mask_dir in zip(config_param.SUBSAMPLE_IMAGE_DIR, config_param.SUBSAMPLE_MASK_DIR):
                images, masks = CalperumDataset.load_subsampled_data(image_dir, mask_dir)
                all_images.extend(images)
                all_masks.extend(masks)
            
            # Initialize the dataset with the in-memory data
            dataset = CalperumDataset(transform=config_param.DATA_TRANSFORM, in_memory_data=(all_images, all_masks))
            
            # Create subsampled indices
            all_subsampled_indices = list(range(len(all_images)))

        else:
            logging.info("No subsampled image and mask files found in the specified directories. Performing subsampling and saving data...")
            
            # Perform subsampling and save the data for each pair of directories
            for image_dir, mask_dir in zip(config_param.SUBSAMPLE_IMAGE_DIR, config_param.SUBSAMPLE_MASK_DIR):
                dataset, subsampled_indices = get_dataset_splits(
                    image_folder=config_param.IMAGE_FOLDER, 
                    mask_folder=config_param.MASK_FOLDER, 
                    transform=config_param.DATA_TRANSFORM, 
                    soil_threshold=50.0, 
                    soil_class=0, 
                    removal_ratio=0.5, 
                    num_classes=config_param.OUT_CHANNELS
                )
                
                # Load the newly created subsampled data
                images, masks = CalperumDataset.load_subsampled_data(image_dir, mask_dir)
                all_images.extend(images)
                all_masks.extend(masks)
                all_subsampled_indices.extend(subsampled_indices)
            
            # Re-initialize the dataset with the in-memory data
            dataset = CalperumDataset(transform=config_param.DATA_TRANSFORM, in_memory_data=(all_images, all_masks))

        if dataset is None:
            raise ValueError("Dataset is None. Please check the dataset initialization.")

        test_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, shuffle=False, num_workers=config_param.NUM_WORKERS)

        # Initialize the model
        model = UNetModule().to(config_param.DEVICE)

        # Load the saved model state
        # model_checkpoint_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low/block_2_epoch_96.pth' #low  
        # model_checkpoint_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low/block_3_epoch_108.pth' #low  (96, '108')
        # model_checkpoint_path= '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium/block_2_epoch_55.pth' #medium 
        # model_checkpoint_path= '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium/block_3_epoch_117.pth' #medium ('55', 117)
        model_checkpoint_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense/block_3_epoch_105.pth' #dense
        
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=config_param.DEVICE))
        model.eval()
        logging.info(f"Model loaded from {model_checkpoint_path}")

        # Run inference on the large area
        run_large_area_inference(model, img_path, save_dir, tile_size=tile_size, device=config_param.DEVICE)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()




# %%
############ 12.c --> improve with subsample indices
import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from dataset.data_loaders_fold_blockcross_subsampling import (
    combine_and_process_paths, get_dataset_splits, save_subsampled_data, plot_with_coordinates, extract_coordinates,
    block_cross_validation
)

def main():
    try:
        # Environment setup for debugging
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # Define parameters
        # output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low' #low
        output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium' #medium
        # output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense' #dense
        os.makedirs(output_dir, exist_ok=True)
        
        # img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif' #low  (30, '33')
        img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.22.tif' # medium ('22', 26)
        # img_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/stacked/tiles_multispectral.118.tif'  # dense (30, '118')
     
        tile_size = 256  # Size of the tiles
        
        # save_dir = 'wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_96'  # Directory to save the predictions low
        save_dir = 'wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_55'  # Directory to save the predictions medium
        # save_dir = 'wombat_predictions_stitch_low_1024_120ep_raw_bestmodel_105'  # Directory to save the predictions dense
        
        indices_save_path = os.path.join(output_dir, 'subsampled_indices.json')

        # Check if subsampled indices and data exist
        if os.path.exists(indices_save_path):
            print(f"Loading subsampled indices from {indices_save_path}...")

            with open(indices_save_path, 'r') as f:
                subsampled_indices = json.load(f)

            # Load subsampled images and masks
            images, masks = CalperumDataset.load_subsampled_data(config_param.SUBSAMPLE_IMAGE_DIR, config_param.SUBSAMPLE_MASK_DIR)
            dataset = CalperumDataset(transform=config_param.DATA_TRANSFORM, in_memory_data=(images, masks))
        else:
            raise FileNotFoundError("Subsampled indices file not found. Please ensure subsampling has been completed and the indices are saved.")

        # Create DataLoader for the dataset
        test_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, shuffle=False, num_workers=config_param.NUM_WORKERS)

        # Initialize the model
        model = UNetModule().to(config_param.DEVICE)
        
        # Load the saved model state
        # model_checkpoint_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low/block_2_epoch_96.pth' #low ('105')
        model_checkpoint_path= '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium/block_2_epoch_55.pth' #medium
        # model_checkpoint_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense/block_3_epoch_105.pth' #dense
        
        model.eval()
        print(f"Model loaded from {model_checkpoint_path}")
        
        # Run inference on the large area
        run_large_area_inference(model, img_path, save_dir, tile_size=tile_size, device=config_param.DEVICE)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
