# --- REBUILD FOLDS TO INCLUDE AUGMENTED DATA ---
block_cv_splits = block_cross_validation(
    dataset=dataset,
    combined_data=[combined_data[i % len(combined_data)] for i in range(len(dataset))],
    num_blocks=config_param.NUM_BLOCKS,
    kmeans_centroids=centroids
)# --- REBUILD FOLDS TO INCLUDE AUGMENTED DATA ---
block_cv_splits = block_cross_validation(
    dataset=dataset,
    combined_data=[combined_data[i % len(combined_data)] for i in range(len(dataset))],
    num_blocks=config_param.NUM_BLOCKS,
    kmeans_centroids=centroids
)import os
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import DataLoader, SubsetRandomSampler
from model.unet_module import UNetModule

from dataset.image_preprocessing import load_raw_multispectral_image
from dataset.calperum_dataset import CalperumDataset
from dataset.data_loaders_fold_blockcross_subsampling import (
    combine_and_process_paths, get_dataset_splits, save_subsampled_data, plot_with_coordinates, extract_coordinates,
    block_cross_validation
)

from metrics.evaluation_bestmodel import ModelEvaluator, initialize_all_metrics 
from metrics.loss_function_loop_blockcross_bestmodel import run_training_loop
from metrics.loss_functions import FocalLoss
import config_param

from rasterio.transform import Affine
from map.plot_blocks_folds import plot_blocks_folds 

import json
from sklearn.cluster import KMeans
import json
from torchmetrics.classification import ConfusionMatrix
import pandas as pd
from collections import Counter
from dataset.data_augmentation import augment_minority_classes_pixel_level, apply_combined_augmentations, augment_minority_classes
from balance_mask_water import integrate_water_distribution
import random

def print_gpu_memory_usage(stage=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    cached = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"{stage} - Allocated memory: {allocated:.2f} GB, Cached memory: {cached:.2f}")
    
    
def log_message(message, log_file):
    log_directory = os.path.dirname(log_file)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)
    
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')


def setup_logging_and_checkpoints():
    ## logs outputs site-specific-models:
    # tb_logs_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low' #low
    tb_logs_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium' #medium
    # tb_logs_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense' #dense    
    os.makedirs(tb_logs_path, exist_ok=True)
    logger = TensorBoardLogger(save_dir=tb_logs_path, name="UNetModel_5b_v100")
    
    os.makedirs(config_param.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config_param.CHECKPOINT_DIR,
        filename='unet_segmentation_{epoch:02d}_{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    return logger, checkpoint_callback


def setup_model_and_optimizer():
    model = UNetModule().to(config_param.DEVICE)
    optimizer = config_param.OPTIMIZER(
        model.parameters(), 
        lr=config_param.LEARNING_RATE, 
        betas=(0.9, 0.999), 
        weight_decay=config_param.WEIGHT_DECAY
    )
    criterion = config_param.CRITERION
    return model, optimizer, criterion

# def setup_model_and_optimizer(criterion):
#     model = UNetModule(criterion).to(config_param.DEVICE)
#     optimizer = config_param.OPTIMIZER(
#         model.parameters(), 
#         lr=config_param.LEARNING_RATE, 
#         betas=(0.9, 0.999), 
#         weight_decay=config_param.WEIGHT_DECAY
#     )
#     return model, optimizer, criterion

def save_model(model, model_checkpoint_path):
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"Model saved to {model_checkpoint_path}")


def save_loss_metrics(train_losses, val_losses, output_dir):
    metrics_file_path = os.path.join(output_dir, 'loss_metrics.txt')
    with open(metrics_file_path, 'w') as f:
        for block_idx, (train_loss_list, val_loss_list) in enumerate(zip(train_losses, val_losses)):
            avg_train_loss_block = np.mean(train_loss_list)
            avg_val_loss_block = np.mean(val_loss_list)
            
            f.write(f"Block {block_idx + 1} Loss Metrics:\n")
            f.write(f"Training Losses per Epoch: {', '.join([f'{loss:.4f}' for loss in train_loss_list])}\n")
            f.write(f"Validation Losses per Epoch: {', '.join([f'{loss:.4f}' for loss in val_loss_list])}\n")
            f.write(f"Average Training Loss: {avg_train_loss_block:.4f}\n")
            f.write(f"Average Validation Loss: {avg_val_loss_block:.4f}\n\n")

        avg_train_loss_across_blocks = np.mean([np.mean(train_loss_list) for train_loss_list in train_losses])
        avg_val_loss_across_blocks = np.mean([np.mean(val_loss_list) for val_loss_list in val_losses])
        
        f.write("Average Loss Across All Blocks:\n")
        f.write(f"Overall Average Training Loss: {avg_train_loss_across_blocks:.4f}\n")
        f.write(f"Overall Average Validation Loss: {avg_val_loss_across_blocks:.4f}\n")

    print(f"Loss metrics saved at {metrics_file_path}")


def save_average_loss_plot(train_losses, val_losses, max_epochs, output_dir):
    avg_train_loss_per_epoch = np.mean(train_losses, axis=0)
    avg_val_loss_per_epoch = np.mean(val_losses, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), avg_train_loss_per_epoch, color='#fc3468', label='Average Train Loss Across All Blocks')
    plt.plot(range(1, max_epochs + 1), avg_val_loss_per_epoch, color='blue', label='Average Validation Loss Across All Blocks')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Epoch vs. Average Loss Across All Blocks')
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join(output_dir, 'average_training_validation_loss_plot_across_blocks.png')
    plt.savefig(plot_file)
    plt.show()
    print(f"Average loss plot across all blocks saved as {plot_file}")

def save_final_model_metrics(metrics, block_idx, output_dir):
    final_metrics_file_path = os.path.join(output_dir, f'final_model_metrics_block_{block_idx + 1}.txt')
    with open(final_metrics_file_path, 'w') as f:
        f.write(f"Final Model Metrics for Block {block_idx + 1} (After Last Epoch):\n")
        f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Precision: {', '.join([f'{p:.4f}' for p in metrics.get('precision', [])])}\n")
        f.write(f"Recall: {', '.join([f'{r:.4f}' for r in metrics.get('recall', [])])}\n")
        f.write(f"F1 Score: {', '.join([f'{f1:.4f}' for f1 in metrics.get('f1', [])])}\n")
        f.write(f"IoU: {', '.join([f'{iou:.4f}' for iou in metrics.get('iou', [])])}\n")
        f.write(f"MIoU: {metrics.get('miou', 0):.4f}\n")
        f.write(f"Unique classes: {', '.join(map(str, metrics.get('unique_classes', [])))}\n")
        f.write(f"Counts per class (%): {', '.join([f'{p:.2f}%' for p in metrics.get('counts_per_class_percentage', [])])}\n")
    print(f"Final model metrics for Block {block_idx + 1} saved at: {final_metrics_file_path}")


def save_best_model_metrics(metrics, block_idx, output_dir):
    best_metrics_file_path = os.path.join(output_dir, f'best_model_metrics_block_{block_idx + 1}.txt')
    
    with open(best_metrics_file_path, 'w') as f:
        f.write(f"Best Model Metrics for Block {block_idx + 1}:\n")
        f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Precision: {', '.join([f'{p:.4f}' for p in metrics.get('precision', [])])}\n")
        f.write(f"Recall: {', '.join([f'{r:.4f}' for r in metrics.get('recall', [])])}\n")
        f.write(f"F1 Score: {', '.join([f'{f1:.4f}' for f1 in metrics.get('f1', [])])}\n")
        f.write(f"IoU: {', '.join([f'{iou:.4f}' for iou in metrics.get('iou', [])])}\n")
        f.write(f"MIoU: {metrics.get('miou', 0):.4f}\n")
        f.write(f"Unique Classes: {', '.join(map(str, metrics.get('unique_classes', [])))}\n")
        f.write(f"Counts per Class (%): {', '.join([f'{count:.2f}%' for count in metrics.get('counts_per_class_percentage', [])])}\n")
    
    print(f"Best model metrics for Block {block_idx + 1} saved at: {best_metrics_file_path}")


def convert_ndarray_to_list(obj):
    """Recursively convert NumPy arrays and scalars in the object to lists or native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj

def save_validation_metrics(metrics, block_idx, output_dir):
    # Convert any NumPy arrays to lists
    metrics = convert_ndarray_to_list(metrics)

    val_metrics_path = os.path.join(output_dir, f'block_{block_idx + 1}_val_metrics.json')
    with open(val_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Validation metrics for Block {block_idx + 1} saved to {val_metrics_path}")


def save_best_validation_metrics(metrics, block_idx, output_dir):
    # Convert any NumPy arrays and scalars to native Python types
    metrics = convert_ndarray_to_list(metrics)

    best_val_metrics_path = os.path.join(output_dir, f'block_{block_idx + 1}_best_val_metrics.json')
    with open(best_val_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Best validation metrics for Block {block_idx + 1} saved to {best_val_metrics_path}")


def has_water_class(masks, water_class=4):
    for mask in masks:
        if (np.array(mask) == water_class).any():
            return True
    return False


def get_class_distribution(masks, class_labels_dict):
    all_labels = []
    for mask in masks:
        mask_np = np.array(mask).flatten()
        mask_np = mask_np[mask_np != -1]
        mask_np = np.round(mask_np).astype(int)
        all_labels.extend(mask_np)
    counts = Counter(all_labels)
    total = sum(counts.values())
    dist = {}
    for k, v in counts.items():
        class_name = class_labels_dict.get(k, f"Unknown ({k})")
        dist[class_name] = f"{(v/total)*100:.2f}%"
    return dist

def print_class_distribution_per_fold(folds, class_labels_dict, stage=""):
    print(f"\n📦 Class Distribution Per Fold {stage}")
    for i, (train_loader, val_loader, test_loader) in enumerate(folds):
        train_masks = [sample[1] for sample in train_loader.dataset]
        val_masks = [sample[1] for sample in val_loader.dataset]
        test_masks = [sample[1] for sample in test_loader.dataset]
        print(f"\nFold {i+1}")
        print("  Train class distribution:", get_class_distribution(train_masks, class_labels_dict))
        print("  Val class distribution:", get_class_distribution(val_masks, class_labels_dict))
        print("  Test class distribution:", get_class_distribution(test_masks, class_labels_dict))

def print_overall_class_distribution(masks, class_labels_dict, stage=""):
    dist = get_class_distribution(masks, class_labels_dict)
    print(f"\nOverall Class Distribution {stage}:")
    for k, v in dist.items():
        print(f"  {k}: {v}")


def print_sample_counts_per_fold(folds, stage=""):
    print(f"\nSample counts per fold {stage}:")
    for i, (train_loader, val_loader, test_loader) in enumerate(folds):
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f"  Block {i+1}:")
        print(f"    Train size: {train_size}")
        print(f"    Val size:   {val_size}")
        print(f"    Test size:  {test_size}")  # <-- Added line

# --- Compute alpha for FocalLoss after water redistribution ---
# def compute_class_weights_from_masks(masks, num_classes=5, ignore_index=-1):
#     import numpy as np
#     all_labels = []
#     for mask in masks:
#         mask_np = np.array(mask).flatten()
#         mask_np = mask_np[mask_np != ignore_index]
#         all_labels.extend(mask_np)
#     counts = np.bincount(all_labels, minlength=num_classes)
#     freq = counts / np.sum(counts)
#     freq = np.clip(freq, 1e-8, 1.0)  # Avoid zeros
#     inv = 1 / freq
#     alpha = inv / np.sum(inv)
#     return alpha.tolist()

def hybrid_block_stratified_cv(dataset, combined_data, num_blocks, kmeans_centroids, class_labels_dict):
    coordinates = extract_coordinates(combined_data)
    kmeans = KMeans(n_clusters=num_blocks, init='k-means++', random_state=42).fit(coordinates)
    block_labels = kmeans.labels_

    folds = []
    for block_id in range(num_blocks):
        block_indices = np.where(block_labels == block_id)[0]
        block_masks = [dataset.masks[i] for i in block_indices]
        # Build multi-label presence matrix
        multi_labels = np.array([[(mask == i).any() for i in range(config_param.OUT_CHANNELS)] for mask in block_masks])
        sss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(sss.split(block_indices, multi_labels))
        # Further split train_val_idx for val
        sss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        train_idx, val_idx = next(sss2.split(block_indices[train_val_idx], multi_labels[train_val_idx]))
        train_indices = block_indices[train_val_idx][train_idx]
        val_indices = block_indices[train_val_idx][val_idx]
        test_indices = block_indices[test_idx]

        train_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
        val_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, sampler=SubsetRandomSampler(val_indices))
        test_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, sampler=SubsetRandomSampler(test_indices))
        folds.append((train_loader, val_loader, test_loader))

    return folds


# def downsample_majority_class(dataset, class_labels_dict, majority_class_name, max_ratio=0.20, min_samples=200):
#     """
#     Downsample the majority class in the dataset to a maximum ratio.
#     Returns new images, masks, and indices kept.
#     """
#     majority_class_idx = [k for k, v in class_labels_dict.items() if v == majority_class_name][0]
#     all_indices = list(range(len(dataset.images)))
#     majority_indices = [i for i in all_indices if (np.array(dataset.masks[i]) == majority_class_idx).sum() > 0]
#     minority_indices = [i for i in all_indices if i not in majority_indices]

#     total = len(all_indices)
#     desired_majority_count = max(int(max_ratio * total), min_samples)
#     if len(majority_indices) > desired_majority_count:
#         keep_majority_indices = random.sample(majority_indices, desired_majority_count)
#     else:
#         keep_majority_indices = majority_indices

#     final_indices = keep_majority_indices + minority_indices
#     random.shuffle(final_indices)
#     new_images = [dataset.images[i] for i in final_indices]
#     new_masks = [dataset.masks[i] for i in final_indices]
#     return new_images, new_masks, final_indices

def mask_be_pixels(masks, be_class_idx, replace_with=-1):
    new_masks = []
    for mask in masks:
        mask_arr = np.array(mask)
        mask_arr[mask_arr == be_class_idx] = replace_with
        new_masks.append(mask_arr)
    return new_masks

def reduce_dominant_class_pixels(masks, class_labels_dict, reduce_fraction=0.5, rare_class_names=('SI', 'WI')):
    # Find dominant class by pixel count across all masks
    pixel_counts = {k: 0 for k in class_labels_dict.keys()}
    for mask in masks:
        mask_np = np.array(mask).flatten()
        mask_np = mask_np[mask_np != -1]
        for k in pixel_counts:
            pixel_counts[k] += np.sum(mask_np == k)
    dominant_class_idx = max(pixel_counts, key=pixel_counts.get)
    dominant_class_name = class_labels_dict[dominant_class_idx]
    print(f"Reducing pixels of dominant class '{dominant_class_name}' (index {dominant_class_idx}) in all masks by {int(reduce_fraction*100)}%.")

    # Get indices of rare classes
    rare_class_indices = [k for k, v in class_labels_dict.items() if v in rare_class_names]

    new_masks = []
    for mask in masks:
        mask_arr = np.array(mask)
        dom_pixels = np.where(mask_arr == dominant_class_idx)
        n_dom = len(dom_pixels[0])
        n_reduce = int(n_dom * reduce_fraction)
        if n_reduce > 0:
            # Find most common non-rare, non-dominant class in this mask
            flat = mask_arr.flatten()
            flat = flat[flat != -1]
            vals, counts = np.unique(flat, return_counts=True)
            valid_mask = np.array([(v != dominant_class_idx) and (v not in rare_class_indices) for v in vals])
            vals_valid = vals[valid_mask]
            counts_valid = counts[valid_mask]
            if len(vals_valid) > 0:
                replacement_class = vals_valid[np.argmax(counts_valid)]
            else:
                replacement_class = dominant_class_idx  # fallback
            # Randomly select pixels to change
            idxs = np.random.choice(range(n_dom), size=n_reduce, replace=False)
            mask_arr[dom_pixels[0][idxs], dom_pixels[1][idxs]] = replacement_class
        new_masks.append(mask_arr)
    return new_masks, dominant_class_name

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # After importing config_param
    inv_class_labels = {v: k for k, v in config_param.class_labels.items()}
    class_labels_dict = {i: inv_class_labels[i] for i in range(config_param.OUT_CHANNELS)}
    print("Class labels dict:", class_labels_dict)
    # class_labels_dict = {0: 'BE', 1: 'NPV', 2: 'PV', 3: 'SI', 4: 'WI'}
    logger, checkpoint_callback = setup_logging_and_checkpoints()
    
    ## outputs site-specific-models:
    # output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low' #low
    # output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium' #medium
    # output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense'#dense
    output_dir = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense' #dense
    os.makedirs(output_dir, exist_ok=True)

    image_dirs = config_param.IMAGE_FOLDER
    mask_dirs = config_param.MASK_FOLDER
    
    combined_data = combine_and_process_paths(image_dirs, mask_dirs)
 
    indices_save_path = os.path.join(output_dir, 'subsampled_indices.json')
    combined_indices_save_path = os.path.join(output_dir, 'combined_indices.json')  # Define the combined indices save path
    
    subsampled_image_exists = all(
        os.path.exists(dir) and os.listdir(dir) 
        for dir in config_param.SUBSAMPLE_IMAGE_DIR
    )
    subsampled_mask_exists = all(
        os.path.exists(dir) and os.listdir(dir)
        for dir in config_param.SUBSAMPLE_MASK_DIR
    )

    if subsampled_image_exists and subsampled_mask_exists and os.path.exists(indices_save_path):
        print(f"Subsampled image and mask files found in {config_param.SUBSAMPLE_IMAGE_DIR} and {config_param.SUBSAMPLE_MASK_DIR}, loading data...")
        
        images, masks = CalperumDataset.load_subsampled_data(config_param.SUBSAMPLE_IMAGE_DIR, config_param.SUBSAMPLE_MASK_DIR)
        
        with open(indices_save_path, 'r') as f:
            subsampled_indices = json.load(f)
            
        dataset = CalperumDataset(transform=config_param.DATA_TRANSFORM, in_memory_data=(images, masks))
        
    else:
        print("No subsampled image and mask files found. Performing subsampling and saving data...")
        
        dataset, subsampled_indices, subsampled_images, subsampled_masks = get_dataset_splits(
            image_folder=image_dirs, 
            mask_folder=mask_dirs, 
            combined_data=combined_data, 
            transform=config_param.DATA_TRANSFORM, 
            soil_threshold=50.0, 
            soil_class=0, 
            removal_ratio=0.5, 
            num_classes=config_param.OUT_CHANNELS,
            indices_save_path=indices_save_path
        )
        # Print or log subsampled indices
        print("Subsampled Indices:", subsampled_indices)
        
        save_subsampled_data(
            subsampled_images=subsampled_images, 
            subsampled_masks=subsampled_masks, 
            combined_data=combined_data, 
            subsampled_indices=subsampled_indices, 
            image_subsample_dir=config_param.SUBSAMPLE_IMAGE_DIR, 
            mask_subsample_dir=config_param.SUBSAMPLE_MASK_DIR,
            indices_save_path=indices_save_path,
            combined_indices_save_path=combined_indices_save_path
        )
        
        images, masks = CalperumDataset.load_subsampled_data(config_param.SUBSAMPLE_IMAGE_DIR, config_param.SUBSAMPLE_MASK_DIR)
        dataset = CalperumDataset(transform=config_param.DATA_TRANSFORM, in_memory_data=(images, masks))

    # Perform KMeans on original data and save centroids
    coordinates = extract_coordinates(combined_data)  # Assume extract_coordinates is a function to get coordinates
    kmeans = KMeans(n_clusters=config_param.NUM_BLOCKS, init='k-means++', random_state=42).fit(coordinates)
    centroids = kmeans.cluster_centers_

    original_data_log_file = os.path.join(output_dir, 'original_data_logfile.txt')
    plot_with_coordinates(dataset, combined_data, log_file_path=original_data_log_file, num_blocks=config_param.NUM_BLOCKS)

    subsampled_data_log_file = os.path.join(output_dir, 'subsampled_data_logfile.txt')
    plot_with_coordinates(dataset, combined_data, indices=subsampled_indices, log_file_path=subsampled_data_log_file, num_blocks=config_param.NUM_BLOCKS)
    
    print(f"Subsampled Indices Length: {len(subsampled_indices)}")
    print(f"Number of Blocks: {config_param.NUM_BLOCKS}")
    print_overall_class_distribution(masks, class_labels_dict, "BEFORE Augmentation")
    
    # --- CONDITIONAL DATA AUGMENTATION BLOCK ---
    # Get water class index from config
    water_class_name = 'WI'
    water_class_idx = config_param.class_labels[water_class_name]
    # Check if water is present
    water_present = has_water_class(masks, water_class=water_class_idx)
    # water_present = has_water_class(masks, water_class=4)
   
    # Calculate initial class distribution
    overall_class_distribution = {}
    all_labels = []
    for mask in masks:
        mask_np = np.array(mask).flatten()
        mask_np = mask_np[mask_np != -1]
        mask_np = np.round(mask_np).astype(int)
        all_labels.extend(mask_np)
    counts = Counter(all_labels)
    total = sum(counts.values())
    for k, v in counts.items():
        class_name = class_labels_dict.get(k, f"Unknown ({k})")
        overall_class_distribution[class_name] = (v / total) * 100


    # --- BLOCK CROSS-VALIDATION SPLITS (before augmentation, for fold assignments) ---
    block_cv_splits = block_cross_validation(
        dataset=dataset,
        combined_data=[combined_data[i] for i in subsampled_indices],
        num_blocks=config_param.NUM_BLOCKS,
        kmeans_centroids=centroids
    )

    # Assign each sample to its fold
    fold_assignments = {}
    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(block_cv_splits):
        for idx in train_loader.dataset.indices:
            fold_assignments[idx] = 'train'
        for idx in val_loader.dataset.indices:
            fold_assignments[idx] = 'val'
        for idx in test_loader.dataset.indices:
            fold_assignments[idx] = 'test'

    # Print class distribution per fold BEFORE augmentation
    print_class_distribution_per_fold(block_cv_splits, class_labels_dict, "BEFORE Augmentation")
    print_sample_counts_per_fold(block_cv_splits, stage="BEFORE Augmentation")

    # --- DATA AUGMENTATION ---
    if config_param.ENABLE_DATA_AUGMENTATION:
        print("Data augmentation is ENABLED.")
        print(f"Dataset size before augmentation: {len(dataset.images)}")
        MIN_CLASS_RATIO = 0.20  # 20% minimum for any class

        percentages = list(overall_class_distribution.values())
        max_percent = max(percentages)
        max_class = max(overall_class_distribution, key=overall_class_distribution.get)

        frequencies = np.array([overall_class_distribution[c] for c in class_labels_dict.values()]) / 100.0

        min_target = np.percentile(frequencies, 10)
        max_target = np.percentile(frequencies, 90)
        min_target = max(0.10, min_target * 0.8)
        max_target = min(0.40, max_target * 1.2)

        mean = np.mean(frequencies)      # <-- Add this
        std = np.std(frequencies)        # <-- And this

        def gaussian_target(freq, mean, std, min_target, max_target):
            z = (freq - mean) / (std + 1e-8)
            score = 1 / (1 + np.exp(z))  # Lower freq -> higher target
            return min_target + (max_target - min_target) * score

        target_ratios = {}
        for i, class_name in enumerate(class_labels_dict.values()):
            freq = frequencies[i]
            target_ratios[class_name] = float(gaussian_target(freq, mean, std, min_target, max_target))

        # Find the class with the highest original frequency
        majority_class = max(overall_class_distribution, key=overall_class_distribution.get)
        # Robust adaptive penalty for the majority class
        majority_ratio = overall_class_distribution[majority_class] / 100.0
        mean_ratio = np.mean(list(overall_class_distribution.values())) / 100.0
        penalty_factor = max(0.5, 1.0 - (majority_ratio - mean_ratio) * 2)
        penalty_factor = min(penalty_factor, 0.8)
        print(f"Adaptive penalty_factor for majority class '{majority_class}': {penalty_factor:.2f}")
        target_ratios[majority_class] *= penalty_factor
        
        # Explicitly penalize BE, regardless of whether it's the majority class
        # be_class_name = "BE"
        # if be_class_name in target_ratios:
        #     be_ratio = overall_class_distribution[be_class_name] / 100.0
        #     mean_ratio = np.mean(list(overall_class_distribution.values())) / 100.0
        #     penalty_factor = max(0.3, 1.0 - (be_ratio - mean_ratio) * 2.5)  # Stronger penalty, min 0.3
        #     penalty_factor = min(penalty_factor, 0.8)
        #     print(f"Adaptive penalty_factor for BE: {penalty_factor:.2f}")
        #     target_ratios[be_class_name] *= penalty_factor
        # --- Automatically detect rare classes and set their target ratio ---
        RARE_CLASS_THRESHOLD = 0.15  # Any class with <15% frequency is considered rare
        RARE_CLASS_RATIO = 0.18      # Target ratio for rare classes

        # Find rare classes based on original distribution
        rare_classes = [cls for cls, freq in overall_class_distribution.items() if freq / 100.0 < RARE_CLASS_THRESHOLD]

        for cls in rare_classes:
            if cls in target_ratios:
                target_ratios[cls] = RARE_CLASS_RATIO

        print(f"Set rare class ratios for: {', '.join(rare_classes)} to {RARE_CLASS_RATIO:.2f}")
        # Optional: Renormalize so all ratios sum to 1
        total = sum(target_ratios.values())
        for k in target_ratios:
            target_ratios[k] /= total

        print("Target ratios (Gaussian-based, penalized majority):")
        for k, v in target_ratios.items():
            print(f"  {k}: {v:.3f}")

        # Use apply_combined_augmentations for all classes
        augmented_counts = augment_minority_classes(
            dataset=dataset,
            class_distributions=overall_class_distribution,  
            class_labels=class_labels_dict,
            target_ratios=target_ratios,
            fold_assignments=fold_assignments,
            augmentation_functions=[lambda pair: apply_combined_augmentations(pair[0], pair[1])]
        )

        print(f"Dataset size after augmentation: {len(dataset.images)}")
        print(f"Augmented counts: {augmented_counts}")
        masks = dataset.masks  # Update masks after augmentation

        # --- MASK BE PIXELS IN ALL PATCHES ---
        # ... after augmentation, before fold assignment ...
        dataset.masks, reduced_class_name = reduce_dominant_class_pixels(dataset.masks, class_labels_dict, reduce_fraction=0.5)
        print(f"Reduced pixels of dominant class: {reduced_class_name}")
        masks = dataset.masks  # Update for downstream reporting
    else:
        print("Data augmentation is DISABLED.")

    # --- REBUILD FOLDS TO INCLUDE AUGMENTED DATA ---
    # block_cv_splits = block_cross_validation(
    #     dataset=dataset,
    #     combined_data=[combined_data[i % len(combined_data)] for i in range(len(dataset))],
    #     num_blocks=config_param.NUM_BLOCKS,
    #     kmeans_centroids=centroids
    # )

    # Re-assign fold assignments for all samples (including augmented)
    fold_assignments = {}
    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(block_cv_splits):
        for idx in train_loader.dataset.indices:
            fold_assignments[idx] = 'train'
        for idx in val_loader.dataset.indices:
            fold_assignments[idx] = 'val'
        for idx in test_loader.dataset.indices:
            fold_assignments[idx] = 'test'

    # Print class distribution per fold AFTER augmentation (before water redistribution)
    print_class_distribution_per_fold(block_cv_splits, class_labels_dict, "AFTER Augmentation (before water redistribution)")
    print_overall_class_distribution(masks, class_labels_dict, "AFTER Augmentation (before water redistribution)")
    print_sample_counts_per_fold(block_cv_splits, stage="AFTER Augmentation (before water redistribution)")

    # --- CONDITIONAL WATER REDISTRIBUTION ---
    if water_present and config_param.ENABLE_WATER_REDISTRIBUTION:
        print("Water class detected and water redistribution is ENABLED.")
        final_folds, _ = integrate_water_distribution(
            dataset, masks, block_cv_splits, config_param.NUM_BLOCKS, config_param.BATCH_SIZE, config_param.NUM_WORKERS
        )
    else:   
        print("Water redistribution is DISABLED or no water class present.")
     # After augmentation, assign all new samples to 'train'
    num_original = len(fold_assignments)
    num_total = len(dataset.images)
    for idx in range(num_original, num_total):
        fold_assignments[idx] = 'train'        
        train_indices = [idx for idx, fold in fold_assignments.items() if fold == 'train']
        val_indices = [idx for idx, fold in fold_assignments.items() if fold == 'val']
        test_indices = [idx for idx, fold in fold_assignments.items() if fold == 'test']
        
        train_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
        val_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, sampler=SubsetRandomSampler(val_indices))
        test_loader = DataLoader(dataset, batch_size=config_param.BATCH_SIZE, sampler=SubsetRandomSampler(test_indices))
        
        final_folds = [(train_loader, val_loader, test_loader)]
        print("Water redistribution is DISABLED or no water class present.")
        final_folds = block_cv_splits

    # Print coordinates shape after water redistribution (FIXED)
    coordinates_water = extract_coordinates([combined_data[i % len(combined_data)] for i in range(len(dataset))])
    print(f"Coordinates shape after reshaping (after water redistribution): {np.array(coordinates_water).shape}")

    # alpha = compute_class_weights_from_masks(masks, num_classes=len(class_labels_dict))
    # print("Alpha for FocalLoss (after water redistribution):", alpha)
    # CRITERION = FocalLoss(alpha=alpha, gamma=2, ignore_index=-1)
    # model, optimizer, criterion = setup_model_and_optimizer(CRITERION)
   
    # Print class distribution AFTER water redistribution
    print_class_distribution_per_fold(final_folds, class_labels_dict, "AFTER Water Redistribution")
    print_overall_class_distribution(masks, class_labels_dict, "AFTER Water Redistribution")
    print_sample_counts_per_fold(final_folds, stage="AFTER Water Redistribution")

    # Initialize all required data structures
    all_metrics = initialize_all_metrics(num_blocks=config_param.NUM_BLOCKS)
    conf_matrices = []  # List to store confusion matrices for each block
    all_train_losses = []  # List to store training losses for each block
    all_val_losses = []  # List to store validation losses for each block

    best_model_paths = []  # List to store best model paths for each block
    best_val_losses = []   # List to store best validation losses for each block

    all_preds_across_blocks = []  # To collect predictions across all blocks
    all_trues_across_blocks = []  # To collect ground truth labels across all blocks

    # Initialize lists to collect best model metrics and final model metrics across all blocks
    all_best_model_metrics = []
    all_final_model_metrics = []
    
    # **Initialize structures for validation metrics**
    all_val_metrics = []
    all_best_val_metrics = []

    # Train, validate, and test using cross-validation splits
    for block_idx, (train_loader, val_loader, test_loader) in enumerate(final_folds):
        if train_loader is None or val_loader is None or test_loader is None:
            print(f"Skipping block {block_idx + 1} due to missing data")
            continue
        model, optimizer, criterion = setup_model_and_optimizer()
           
        # Run training loop
        train_losses, val_losses, best_epoch_model_path, best_epoch_val_loss = run_training_loop(
            model, train_loader, val_loader, optimizer, criterion, 
            config_param.NUM_EPOCHS, block_idx, output_dir, config_param.DEVICE, logger
        )
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        
        # Store the best model path and validation loss for the current block
        best_model_paths.append(best_epoch_model_path)
        best_val_losses.append(best_epoch_val_loss)
        
        # Evaluate the final model after the training loop (last epoch model) on the test set
        evaluator = ModelEvaluator(model, test_loader, device=config_param.DEVICE)
        final_metrics = evaluator.run_evaluation(block_idx, all_metrics, conf_matrices)
        all_final_model_metrics.append(final_metrics)
        # Save metrics for the final model on the test set
        save_final_model_metrics(final_metrics, block_idx, output_dir)
        
        # **Confusion Matrix Plotting for Final Model (Test Set)**
        evaluator.plot_confusion_matrix(
            final_metrics['all_preds'], final_metrics['all_trues'],
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'], 
            output_dir=output_dir, block_idx=f"{block_idx}_final_test"
        )
        
        # **Evaluate the final model (last epoch model) on the validation set**
        print(f"Evaluating the final model (last epoch) for Block {block_idx + 1} on the validation set")
        val_evaluator = ModelEvaluator(model, val_loader, device=config_param.DEVICE)      
        # Evaluate the final model on the validation set
        val_final_metrics = val_evaluator.run_evaluation(block_idx, all_metrics, conf_matrices)
        all_val_metrics.append(val_final_metrics)
        # Save metrics for the final model's validation evaluation
        save_validation_metrics(val_final_metrics, block_idx, output_dir)
        
        # **Confusion Matrix Plotting for Final Model (Validation Set)**
        evaluator.plot_confusion_matrix(
            val_final_metrics['all_preds'], val_final_metrics['all_trues'],
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'], 
            output_dir=output_dir, block_idx=f"{block_idx}_final_val"
        )
        
        # **Reload the best model for evaluation on both the test and validation sets**
        print(f"Loading the best model for Block {block_idx + 1} from {best_epoch_model_path}")
        model.load_state_dict(torch.load(best_epoch_model_path))

        # Run evaluation for the best model (test set)
        best_metrics = evaluator.run_evaluation(block_idx, all_metrics, conf_matrices)
        all_best_model_metrics.append(best_metrics)
        # Save metrics for the best model on the test set
        save_best_model_metrics(best_metrics, block_idx, output_dir)
        
        # **Confusion Matrix Plotting for Best Model (Test Set)**
        evaluator.plot_confusion_matrix(
            best_metrics['all_preds'], best_metrics['all_trues'],
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'], 
            output_dir=output_dir, block_idx=f"{block_idx}_best_test"
        )
        
        # ** Run evaluation for the best model (validation set)**
        print(f"Evaluating the best model for Block {block_idx + 1} on the validation set")
        best_val_metrics = val_evaluator.run_evaluation(block_idx, all_metrics, conf_matrices)
        all_best_val_metrics.append(best_val_metrics)      
        # Save metrics for the best model's validation evaluation
        save_best_validation_metrics(best_val_metrics, block_idx, output_dir)
        
        # **Confusion Matrix Plotting for Best Model (Validation Set)**
        evaluator.plot_confusion_matrix(
            best_val_metrics['all_preds'], best_val_metrics['all_trues'],
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'], 
            output_dir=output_dir, block_idx=f"{block_idx}_best_val"
        )
            # **End: Evaluation on Validation Set**
        
        # Extend predictions and ground truths for overall evaluation
        all_preds_across_blocks.extend(best_metrics['all_preds'])
        all_trues_across_blocks.extend(best_metrics['all_trues'])

    # Save average loss plot across all blocks
    save_average_loss_plot(all_train_losses, all_val_losses, config_param.NUM_EPOCHS, output_dir)

    # Save metrics for each epoch
    save_loss_metrics(all_train_losses, all_val_losses, output_dir)

    # **Calculate and Plot Average Metrics and Confusion Matrices**
    # **For Final Models on Test Set**
    avg_metrics_across_final_models_test = evaluator.calculate_average_metrics(all_final_model_metrics)
    if avg_metrics_across_final_models_test:
        print("\nAverage Metrics Across All Final Models (Test Set):")
        print(f"Accuracy: {avg_metrics_across_final_models_test['accuracy']:.4f}")
        print(f"Precision: {', '.join([f'{p:.4f}' for p in avg_metrics_across_final_models_test['precision']])}")
        print(f"Recall: {', '.join([f'{r:.4f}' for r in avg_metrics_across_final_models_test['recall']])}")
        print(f"F1 Score: {', '.join([f'{f1:.4f}' for f1 in avg_metrics_across_final_models_test['f1']])}")
        print(f"IoU: {', '.join([f'{iou:.4f}' for iou in avg_metrics_across_final_models_test['iou']])}")
        print(f"MIoU: {avg_metrics_across_final_models_test['miou']:.4f}")

        evaluator.save_average_metrics(avg_metrics_across_final_models_test, block_idx, output_dir)

        # **Plot Average Confusion Matrix for Final Models on Test Set**
        evaluator.calculate_and_save_average_confusion_matrix(
            conf_matrices,
            all_preds_across_blocks,
            all_trues_across_blocks,
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'],
            output_dir=os.path.join(output_dir, 'final_test_avg')
        )

    # **For Final Models on Validation Set**
    avg_metrics_across_final_models_val = evaluator.calculate_average_metrics(all_val_metrics)
    if avg_metrics_across_final_models_val:
        print("\nAverage Metrics Across All Final Models (Validation Set):")
        print(f"Accuracy: {avg_metrics_across_final_models_val['accuracy']:.4f}")
        print(f"Precision: {', '.join([f'{p:.4f}' for p in avg_metrics_across_final_models_val['precision']])}")
        print(f"Recall: {', '.join([f'{r:.4f}' for r in avg_metrics_across_final_models_val['recall']])}")
        print(f"F1 Score: {', '.join([f'{f1:.4f}' for f1 in avg_metrics_across_final_models_val['f1']])}")
        print(f"IoU: {', '.join([f'{iou:.4f}' for iou in avg_metrics_across_final_models_val['iou']])}")
        print(f"MIoU: {avg_metrics_across_final_models_val['miou']:.4f}")

        evaluator.save_average_metrics(avg_metrics_across_final_models_val, block_idx, output_dir)

        # **Plot Average Confusion Matrix for Final Models on Validation Set**
        evaluator.calculate_and_save_average_confusion_matrix(
            conf_matrices,
            all_preds_across_blocks,
            all_trues_across_blocks,
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'],
            output_dir=os.path.join(output_dir, 'final_val_avg')
        )

    # **For Best Models on Test Set**
    avg_metrics_across_best_models_test = evaluator.calculate_average_metrics(all_best_model_metrics)
    if avg_metrics_across_best_models_test:
        print("\nAverage Metrics Across All Best Models (Test Set):")
        print(f"Accuracy: {avg_metrics_across_best_models_test['accuracy']:.4f}")
        print(f"Precision: {', '.join([f'{p:.4f}' for p in avg_metrics_across_best_models_test['precision']])}")
        print(f"Recall: {', '.join([f'{r:.4f}' for r in avg_metrics_across_best_models_test['recall']])}")
        print(f"F1 Score: {', '.join([f'{f1:.4f}' for f1 in avg_metrics_across_best_models_test['f1']])}")
        print(f"IoU: {', '.join([f'{iou:.4f}' for iou in avg_metrics_across_best_models_test['iou']])}")
        print(f"MIoU: {avg_metrics_across_best_models_test['miou']:.4f}")

        evaluator.save_average_metrics(avg_metrics_across_best_models_test, block_idx, output_dir)

        # **Plot Average Confusion Matrix for Best Models on Test Set**
        evaluator.calculate_and_save_average_confusion_matrix(
            conf_matrices,
            all_preds_across_blocks,
            all_trues_across_blocks,
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'],
            output_dir=os.path.join(output_dir, 'best_test_avg')
        )

    # **For Best Models on Validation Set**
    avg_metrics_across_best_models_val = evaluator.calculate_average_metrics(all_best_val_metrics)
    if avg_metrics_across_best_models_val:
        print("\nAverage Metrics Across All Best Models (Validation Set):")
        print(f"Accuracy: {avg_metrics_across_best_models_val['accuracy']:.4f}")
        print(f"Precision: {', '.join([f'{p:.4f}' for p in avg_metrics_across_best_models_val['precision']])}")
        print(f"Recall: {', '.join([f'{r:.4f}' for r in avg_metrics_across_best_models_val['recall']])}")
        print(f"F1 Score: {', '.join([f'{f1:.4f}' for f1 in avg_metrics_across_best_models_val['f1']])}")
        print(f"IoU: {', '.join([f'{iou:.4f}' for iou in avg_metrics_across_best_models_val['iou']])}")
        print(f"MIoU: {avg_metrics_across_best_models_val['miou']:.4f}")

        evaluator.save_average_metrics(avg_metrics_across_best_models_val, block_idx, output_dir)

        # **Plot Average Confusion Matrix for Best Models on Validation Set**
        evaluator.calculate_and_save_average_confusion_matrix(
            conf_matrices,
            all_preds_across_blocks,
            all_trues_across_blocks,
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'],
            output_dir=os.path.join(output_dir, 'best_val_avg')
        )

    output_log_file = os.path.join(output_dir, "best_model_paths_and_validation_losses.txt")

    # Open the file in append mode
    with open(output_log_file, 'a') as log_file:
        # Print the best model paths and validation losses for all blocks to both console and file
        for block_idx, (best_model_path, best_val_loss) in enumerate(zip(best_model_paths, best_val_losses)):
            output_line = f"Best model for Block {block_idx + 1} saved at: {best_model_path} with validation loss: {best_val_loss:.4f}\n"
            print(output_line)  # Print to console
            log_file.write(output_line)  # Write to file



if __name__ == '__main__':
    main()