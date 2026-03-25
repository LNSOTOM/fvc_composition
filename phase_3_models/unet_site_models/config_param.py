############## Configuration ##########################
import torch
from torch.optim import Adam, RMSprop
import torchvision.transforms as transforms
import re
import os
import random
import numpy as np
from torchvision.transforms import functional as F



from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from metrics.loss_functions import FocalLoss, WeightedCrossEntropyLoss, calculate_class_weights, save_class_weights_to_file, DiceLoss, CombinedDiceFocalLoss

# =========================
# Utility functions (no dataset imports at top-level)
# =========================

def get_num_classes_from_mask(mask_path):
    """Get the unique classes in the mask, excluding NaN values (represented as -1)."""
    # Import inside the function to avoid circular import
    from dataset.calperum_dataset import CalperumDataset
    
    # Load the mask and profile, but ignore the profile
    mask, _ = CalperumDataset.load_mask(mask_path)
    
    # Convert the mask to a tensor
    mask_tensor = torch.tensor(mask)

    # Exclude NaN values (-1) and get unique values
    unique_classes = torch.unique(mask_tensor[mask_tensor != -1])

    # Return the unique classes
    return unique_classes

# def compute_all_unique_classes(mask_folders, class_labels):
#     """Collect all unique (valid) class indices from a list of mask folders."""
#     all_unique_classes = set()
#     for folder in mask_folders:
#         for f in os.listdir(folder):
#             if f.endswith('.tif'):
#                 mask_path = os.path.join(folder, f)
#                 unique_classes = get_num_classes_from_mask(mask_path)
#                 all_unique_classes.update(unique_classes.tolist())
#     # Only keep valid class labels
#     return {int(cls) for cls in all_unique_classes if int(cls) in class_labels.values()}


# ======================
# Define the class labels
# ======================
# class_labels = {'BE': 0, 'NPV': 1, 'PV': 2} #low
# class_labels = {'BE': 0, 'NPV': 1, 'PV': 2, 'SI': 3} #medium
class_labels = {'BE': 0, 'NPV': 1, 'PV': 2, 'SI': 3, 'WI': 4} #dense  



# ======================
# Data directories
# ======================
MASK_FOLDER = [
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/mask_fvc' #low
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/mask_fvc' #medium
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc' #dense
]

IMAGE_FOLDER = [
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/predictors_5b'  #low
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/predictors_5b'  #medium
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/predictors_5b'  #dense
]

# ======================
# Directory for loading or saving raster images + masks files
# ======================
SUBSAMPLE_IMAGE_DIR = [
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/predictors_5b_subsample'  #low
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/low/predictor_5b_subsample'  #sites_low
    
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/predictors_5b_subsample'  #medium
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/medium/predictors_5b_subsample'  #sites_medium
    
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/predictors_5b_subsample'  #dense
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/dense/predictor_5b_subsample'  #sites_dense
]
SUBSAMPLE_MASK_DIR = [
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/low/mask_fvc_subsample'  #low
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/low/mask_fvc_subsample' #sites_low
    
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/medium/mask_fvc_subsample'  #medium
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/medium/mask_fvc_subsample' #sites_medium
    
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample_nowater'
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample' #dense with water
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/dense/mask_fvc_subsample' #sites_dense
]
for directory in SUBSAMPLE_IMAGE_DIR:
    os.makedirs(directory, exist_ok=True)
for directory in SUBSAMPLE_MASK_DIR:
    os.makedirs(directory, exist_ok=True)

AUG_IMAGE_DIR = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/predictor_5b'
AUG_MASK_DIR = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
# AUG_MASK_DIR = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_MASK_DIR, exist_ok=True)

# ======================
# Other paths/configs
# ======================
# CHECKPOINT_DIR = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense'
# CHECKPOINT_DIR = '/home/laura/dense_aug'
CHECKPOINT_DIR = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/medium/original'

# Inference defaults (used by inference_raster_to_geojson.py)
INFERENCE_MODEL_PATH = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth'
INFERENCE_INPUT_RASTER = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.22.tif'
INFERENCE_OUTPUT_GEOJSON = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22/predictions.geojson'
INFERENCE_TILE_SIZE = 3072
INFERENCE_THRESHOLD = 0.5
INFERENCE_POSITIVE_CLASS_ID = 1

# Low-site inference (checkpoint + stacked tiles folder)
INFERENCE_MODEL_PATH_LOW = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/low/original/block_3_epoch_108.pth'
INFERENCE_INPUT_RASTER_LOW = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked'

# Tip: inference_raster_to_geojson.py accepts a directory as --input-raster and will write one GeoJSON per tile
# into the output directory you provide.

INDICES_SAVE_PATHS = [
    # '/home/laura/dense_aug/subsampled_indices.json',
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/medium/original/subsampled_indices.json',
]
COMBINED_INDICES_SAVE_PATHS = [
    # '/home/laura/dense_aug/combined_indices.json',
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_site_models/outputs_ecosystems/medium/original/combined_indices.json',
]

# Initialize a set to track all unique classes found across masks
all_unique_classes = set()

# Iterate through each mask file to collect unique classes - change to MASK_DIR if first time
for folder in SUBSAMPLE_MASK_DIR:  #MASK_DIR
    for f in os.listdir(folder):
        if f.endswith('.tif'):
            mask_path = os.path.join(folder, f)
            unique_classes = get_num_classes_from_mask(mask_path)
            all_unique_classes.update(unique_classes.tolist())

# Ensure we only keep valid class labels (0, 1, 2, 3, 4) and exclude NaN values
all_unique_classes = {int(cls) for cls in all_unique_classes if not np.isnan(cls) and int(cls) in class_labels.values()}


# ======================
# Network parameters
# ======================
IN_CHANNELS = 5

# Call this in your main script if you want dynamic OUT_CHANNELS:
# For consistency, set OUT_CHANNELS to match the number of classes
OUT_CHANNELS = len(all_unique_classes)
print(f"OUT_CHANNELS determined based on unique classes in masks: {OUT_CHANNELS}")

# If you want a fixed OUT_CHANNELS, set manually:
# OUT_CHANNELS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Training Hyperparameters
# ======================
NUM_EPOCHS    = 120
BATCH_SIZE    = 16  ## 4: mini-batch size per GPU step /  #16 --> original models
NUM_WORKERS   = 2   # 2: safe for I/O and CPU usage / #4 or #1 --> original models
OPTIMIZER     = Adam
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
BETAS         = (0.9, 0.999)

CRITERION = FocalLoss(alpha=1, gamma=2, ignore_index=-1)

# Water handling settings
EXCLUDE_WATER = True                # Exclude water during evaluation (set to True to include)
EXCLUDE_WATER_IN_TRAINING = False   # Leave water in during training (set to True to exclude)
WATER_CLASS_INDEX = 4               # The index of the water class (WI)

# ======================
# Data augmentation/transforms
# ======================
APPLY_TRANSFORMS = False

DATA_TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda img_mask: apply_color_jitter(img_mask[0], img_mask[1])),
    transforms.Lambda(lambda img_mask: apply_vertical_flip(img_mask[0], img_mask[1])),
    transforms.Lambda(lambda img_mask: apply_horizontal_flip(img_mask[0], img_mask[1])),
    transforms.Lambda(lambda img_mask: apply_random_affine(img_mask[0], img_mask[1]))
]) if APPLY_TRANSFORMS else None

# ======================
# Miscellaneous
# ======================
NUM_BLOCKS = 3
# ENABLE_WATER_REDISTRIBUTION = False

USE_AUGMENTED_DATA = True #False  True

def combine_and_process_paths(image_dirs, mask_dirs):
    combined_data = []
    for img_dir, mask_dir in zip(image_dirs, mask_dirs):
        region_name = extract_region_name(img_dir)
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
        for index, (img_file, mask_file) in enumerate(zip(img_files, mask_files)):
            full_img_path = os.path.join(img_dir, img_file)
            full_mask_path = os.path.join(mask_dir, mask_file)
            combined_data.append((index, region_name, full_img_path, full_mask_path))
    return combined_data

def extract_region_name(path):
    match = re.search(r'/(low|medium|dense)/', path)
    return match.group(1) if match else "unknown"
