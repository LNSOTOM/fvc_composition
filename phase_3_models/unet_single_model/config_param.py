############## Configuration ##########################
import torch
from torch.optim import Adam, RMSprop

import torchvision.transforms as transforms
import re
import os

import random
import numpy as np
from torchvision.transforms import functional as F

from dataset.calperum_dataset import CalperumDataset
from dataset.data_augmentation import apply_color_jitter, apply_vertical_flip, apply_horizontal_flip, apply_random_affine

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from metrics.loss_functions import FocalLoss, WeightedCrossEntropyLoss, calculate_class_weights, save_class_weights_to_file, DiceLoss, CombinedDiceFocalLoss


def get_num_classes_from_mask(mask_path):
    """Get the unique classes in the mask, excluding NaN values (represented as -1)."""
    # Load the mask and profile, but ignore the profile
    mask, _ = CalperumDataset.load_mask(mask_path)

    # Convert the mask to a tensor
    mask_tensor = torch.tensor(mask)

    # Exclude NaN values (-1) and get unique values
    unique_classes = torch.unique(mask_tensor[mask_tensor != -1])

    # Return the unique classes
    return unique_classes

# Define the class labels
class_labels = {'BE': 0, 'NPV': 1, 'PV': 2, 'SI': 3, 'WI': 4}

MASK_FOLDER = [
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/low/mask_fvc',
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/medium/mask_fvc',
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/dense/mask_fvc'
]

# Initialize a set to track all unique classes found across masks
all_unique_classes = set()

# Iterate through each mask file to collect unique classes
for folder in MASK_FOLDER:
    for f in os.listdir(folder):
        if f.endswith('.tif'):
            mask_path = os.path.join(folder, f)
            unique_classes = get_num_classes_from_mask(mask_path)
            all_unique_classes.update(unique_classes.tolist())

# Ensure we only keep valid class labels (0, 1, 2, 3, 4)
all_unique_classes = {int(cls) for cls in all_unique_classes if int(cls) in class_labels.values()}

# 1.Network parameters
IN_CHANNELS = 5
# Dynamically determine the number of output channels
# Determine the number of output channels based on the maximum number of classes found
OUT_CHANNELS = len(all_unique_classes)
print(f"OUT_CHANNELS determined based on unique classes in masks: {OUT_CHANNELS}")

'''Pytorch Lightning device name = gpu'''
# DEVICE = "gpu" if torch.cuda.is_available() else "cpu" 
'''Pytorch device name = cuda'''
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set the accelerator to "gpu" if CUDA is available, otherwise, set it to "cpu"


# 2.Data handling parameters
##5bands
def combine_and_process_paths(image_dirs, mask_dirs):
    combined_data = []
    for img_dir, mask_dir in zip(image_dirs, mask_dirs):
        # Extract the region name from the directory path
        region_name = extract_region_name(img_dir)

        # List all TIFF files in the image directory
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]

        # Combine paths and index files
        for index, (img_file, mask_file) in enumerate(zip(img_files, mask_files)):
            full_img_path = os.path.join(img_dir, img_file)
            full_mask_path = os.path.join(mask_dir, mask_file)
            combined_data.append((index, region_name, full_img_path, full_mask_path))
    
    return combined_data

def extract_region_name(path):
    # Use a regex to find the region name in the path
    match = re.search(r'/(low|medium|dense)/', path)
    return match.group(1) if match else "unknown"

# Paths to images and masks for multiple sites
IMAGE_FOLDER = [
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/low/predictors_5b',
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/medium/predictors_5b',
    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/dense/predictors_5b'
]


# 3.Hyperparameters for training
'''Num of epochs: how many times the learning algorithm will work through the entire training dataset. Helps to not overfit'''
NUM_EPOCHS = 2 #120 # try also --> 100 and 20 for test and 40 minimum
'''batch_size: number of training samples utilised in one iteration'''
BATCH_SIZE =  16 #12  # minimum 16) | 32 
##PATCH_SIZE = 256  # Used in dataset preprocessing, if applicable
NUM_WORKERS = 4

##5bands
CHECKPOINT_DIR = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/sites' #across sites

#  OPTIMIZER (configuration for loss functions)
'''optimiser: influence model performance'''
OPTIMIZER = Adam      # try also --> RMSprop=Root Mean Squared Propagation  
# LOSS FUNCTIONS - # Optimizer settings
LEARNING_RATE = 1e-4  # 0.0001  or 3e-4  = 3 * 10^-4
'''Weight Decay: provides regularization, helping to prevent overfitting. If you see signs of overfitting 
(e.g., training loss decreasing while validation loss increases), you might consider i'ncreasing the weight decay slightly.'''
WEIGHT_DECAY = 1e-4
'''Betas: are standard settings for Adam, where beta1 controls the decay rate of the running average of the gradient and 
beta2 controls the decay rate of the running average of the squared gradient. 
These values are generally fine and typically don’t need adjustment unless you’re encountering specific issues with convergence.'''
BETAS = (0.9, 0.999)


# Choose a loss function based on your needs:
# Focal Loss parameters
# FOCAL_ALPHA = 1
# FOCAL_GAMMA = 2
# FOCAL_IGNORE_INDEX = -1
# # Dice Loss parameters
# DICE_SMOOTH = 1
# DICE_IGNORE_INDEX = -1
# CRITERION = CombinedDiceFocalLoss(
#     alpha=0.5,          # Balance between focal and dice loss
#     gamma=FOCAL_GAMMA,  # Focal loss focusing parameter
#     smooth=DICE_SMOOTH, # Dice loss smoothing parameter
#     ignore_index=FOCAL_IGNORE_INDEX  # Index to ignore in target mask
# )
# Focal loss:
CRITERION = FocalLoss(alpha=1, gamma=2, ignore_index=-1)  # Now handles NaN values
# OR CRITERION = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, ignore_index=FOCAL_IGNORE_INDEX)

# Define the criterion (DiceLoss) with custom parameters
# CRITERION = DiceLoss(smooth=1, ignore_index=-1)  # Adjust the 'smooth' and 'ignore_index' as needed
# OR CRITERION = DiceLoss(smooth=DICE_SMOOTH, ignore_index=DICE_IGNORE_INDEX)

# CRITERION = WeightedCrossEntropyLoss(weights=class_weights, device=DEVICE)

# CRITERION = CrossEntropyLoss(ignore_index=-1)  # Default to CrossEntropyLoss with ignore_index for NaN values replaced by -1


# 4.Data augmentation - Define a flag to enable or disable transformations
# a) Without Data augmentation
# APPLY_TRANSFORMS = False  # Set to True to apply transformations
# # Define transformations conditionally
# DATA_TRANSFORM = transforms.Compose([
#     transforms.RandomHorizontalFlip(0.5),
# ]) if APPLY_TRANSFORMS else None

# b) Without (set to False)/With Data Augmentation (set to True)
APPLY_TRANSFORMS = False  

# Define transformations conditionally
# DATA_TRANSFORM = transforms.Compose([
#     transforms.Lambda(lambda img: apply_color_jitter(img)),
#     transforms.Lambda(lambda img: apply_vertical_flip(img)),
#     transforms.Lambda(lambda img: apply_horizontal_flip(img)),
#     transforms.Lambda(lambda img: apply_random_affine(img))
# ]) if APPLY_TRANSFORMS else None
DATA_TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda img_mask: apply_color_jitter(img_mask[0], img_mask[1])),
    transforms.Lambda(lambda img_mask: apply_vertical_flip(img_mask[0], img_mask[1])),
    transforms.Lambda(lambda img_mask: apply_horizontal_flip(img_mask[0], img_mask[1])),
    transforms.Lambda(lambda img_mask: apply_random_affine(img_mask[0], img_mask[1]))
]) if APPLY_TRANSFORMS else None


#save Data augmentation
# AUGMENTED_DATA_DIR = '/media/laura/Extreme SSD/code/ecosystem_composition/phase_3_models/unet_model/ecosytems/sites/augmented_data'
# os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)

# # Directory for loading or saving GeoParquet files
# SUBSAMPLE_DATA_DIR = '/media/laura/Extreme SSD/code/ecosystem_composition/phase_3_models/unet_model/ecosytems/sites/subsample_data'
# os.makedirs(SUBSAMPLE_DATA_DIR, exist_ok=True)


# Directory for loading or saving raster images + masks files
SUBSAMPLE_IMAGE_DIR = [
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/low/predictors_5b_subsample',
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/medium/predictors_5b_subsample',
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/dense/predictors_5b_subsample'
]
# os.makedirs(SUBSAMPLE_IMAGE_DIR, exist_ok=True)

SUBSAMPLE_MASK_DIR = [
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/low/mask_fvc_subsample',
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/medium/mask_fvc_subsample',
                    '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/dense/mask_fvc_subsample'
]

# # Single directory for storing combined subsampled images and masks
# SUBSAMPLE_IMAGE_DIR = '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites/predictors_5b_subsample'
# SUBSAMPLE_MASK_DIR = '/media/laura/Extreme SSD/qgis/calperumResearch/unet_single_model_5b/sites/mask_fvc_subsample'
# # Create directories if they don’t exist
# os.makedirs(SUBSAMPLE_IMAGE_DIR, exist_ok=True)
# os.makedirs(SUBSAMPLE_MASK_DIR, exist_ok=True)

# os.makedirs(SUBSAMPLE_MASK_DIR, exist_ok=True)
# Create the directories if they don't exist
for directory in SUBSAMPLE_IMAGE_DIR:
    os.makedirs(directory, exist_ok=True)

for directory in SUBSAMPLE_MASK_DIR:
    os.makedirs(directory, exist_ok=True)

NUM_BLOCKS = 3
# NUM_FOLDS = 5 # Number of splits for GroupKFold

# Define lists of paths to your saved JSON files
INDICES_SAVE_PATHS = [
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/low/subsampled_indices.json',
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/medium/subsampled_indices.json',
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/subsampled_indices.json'
]

COMBINED_INDICES_SAVE_PATHS = [
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/low/combined_indices.json',
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/medium/combined_indices.json',
    '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/combined_indices.json'
]

# # Paths for combined subsample indices
# INDICES_SAVE_PATH = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/sites/subsampled_indices.json'
# COMBINED_INDICES_SAVE_PATH = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/sites/combined_indices.json'


## Display tensorboard
#  tensorboard --logdir=phase_3_models/unet_model/low/tb_logs
