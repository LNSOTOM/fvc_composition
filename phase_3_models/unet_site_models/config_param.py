############## Configuration ##########################
import torch
from torch.optim import Adam, RMSprop
import torchvision.transforms as transforms
import re
import os
import random
import numpy as np
from torchvision.transforms import functional as F

from dataset.data_augmentation import apply_color_jitter, apply_vertical_flip, apply_horizontal_flip, apply_random_affine

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from metrics.loss_functions import FocalLoss, WeightedCrossEntropyLoss, calculate_class_weights, save_class_weights_to_file, DiceLoss, CombinedDiceFocalLoss

# =========================
# Utility functions (no dataset imports at top-level)
# =========================

def get_num_classes_from_mask(mask_path):
    """Get the unique classes in the mask, excluding NaN values (represented as -1)."""
    # Import inside the function to avoid circular import
    from dataset.calperum_dataset import CalperumDataset
    mask = CalperumDataset.load_mask(mask_path)
    mask_tensor = torch.tensor(mask)
    unique_classes = torch.unique(mask_tensor[mask_tensor != -1])
    return unique_classes

def compute_all_unique_classes(mask_folders, class_labels):
    """Collect all unique (valid) class indices from a list of mask folders."""
    all_unique_classes = set()
    for folder in mask_folders:
        for f in os.listdir(folder):
            if f.endswith('.tif'):
                mask_path = os.path.join(folder, f)
                unique_classes = get_num_classes_from_mask(mask_path)
                all_unique_classes.update(unique_classes.tolist())
    # Only keep valid class labels
    return {int(cls) for cls in all_unique_classes if int(cls) in class_labels.values()}

# ======================
# Define the class labels
# ======================
class_labels = {'BE': 0, 'NPV': 1, 'PV': 2, 'SI': 3, 'WI': 4}

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
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample'
    # '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites_freq/dense/mask_fvc_subsample' #sites_dense
]
for directory in SUBSAMPLE_IMAGE_DIR:
    os.makedirs(directory, exist_ok=True)
for directory in SUBSAMPLE_MASK_DIR:
    os.makedirs(directory, exist_ok=True)

AUG_IMAGE_DIR = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/predictor_5b'
AUG_MASK_DIR = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/aug/mask_fvc'
os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_MASK_DIR, exist_ok=True)

# ======================
# Other paths/configs
# ======================
CHECKPOINT_DIR = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense'
INDICES_SAVE_PATHS = [
    '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/subsampled_indices.json'
]
COMBINED_INDICES_SAVE_PATHS = [
    '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/combined_indices.json'
]

# ======================
# Network parameters
# ======================
IN_CHANNELS = 5

# Call this in your main script if you want dynamic OUT_CHANNELS:
# all_unique_classes = compute_all_unique_classes(MASK_FOLDER, class_labels)
# OUT_CHANNELS = len(all_unique_classes)
# print(f"OUT_CHANNELS determined based on unique classes in masks: {OUT_CHANNELS}")

# If you want a fixed OUT_CHANNELS, set manually:
OUT_CHANNELS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Training Hyperparameters
# ======================
NUM_EPOCHS    = 120
BATCH_SIZE    = 16
NUM_WORKERS   = 4

OPTIMIZER     = Adam
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
BETAS         = (0.9, 0.999)

CRITERION = FocalLoss(alpha=1, gamma=2, ignore_index=-1)

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
ENABLE_WATER_REDISTRIBUTION = False
USE_AUGMENTED_DATA = True #False

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
