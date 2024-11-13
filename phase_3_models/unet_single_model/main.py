import os
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model.unet_module import UNetModule

from dataset.image_preprocessing import load_raw_multispectral_image
from dataset.calperum_dataset import CalperumDataset
from dataset.data_loaders_fold_blockcross_subsampling import (
    combine_and_process_paths, get_dataset_splits, save_subsampled_data, plot_with_coordinates, extract_coordinates,
    block_cross_validation
)

from metrics.evaluation_bestmodel import ModelEvaluator, initialize_all_metrics 
from metrics.loss_function_loop_blockcross_bestmodel import run_training_loop
import config_param

from rasterio.transform import Affine
from map.plot_blocks_folds import plot_blocks_folds 

import json
from sklearn.cluster import KMeans
import json
from torchmetrics.classification import ConfusionMatrix
import pandas as pd



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
    tb_logs_path = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/sites' #across sites 
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




def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    logger, checkpoint_callback = setup_logging_and_checkpoints()
    
    output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/sites'
    os.makedirs(output_dir, exist_ok=True)

    image_dirs = config_param.IMAGE_FOLDER
    mask_dirs = config_param.MASK_FOLDER

    # Combine paths from all three folders (low, medium, dense)
    combined_data = combine_and_process_paths(image_dirs, mask_dirs)
    
    indices_save_path = os.path.join(output_dir, 'subsampled_indices.json')
    combined_indices_save_path = os.path.join(output_dir, 'combined_indices.json')
    
    # Check if subsampled images and masks already exist in a single directory
    subsampled_image_exists = os.path.exists(config_param.SUBSAMPLE_IMAGE_DIR) and os.listdir(config_param.SUBSAMPLE_IMAGE_DIR)
    subsampled_mask_exists = os.path.exists(config_param.SUBSAMPLE_MASK_DIR) and os.listdir(config_param.SUBSAMPLE_MASK_DIR)

    # Load or generate subsampled data
    if subsampled_image_exists and subsampled_mask_exists and os.path.exists(indices_save_path):
        print("Loading existing subsampled data...")
        images, masks = CalperumDataset.load_subsampled_data(config_param.SUBSAMPLE_IMAGE_DIR, config_param.SUBSAMPLE_MASK_DIR)
        
        with open(indices_save_path, 'r') as f:
            subsampled_indices = json.load(f)
        
        dataset = CalperumDataset(transform=config_param.DATA_TRANSFORM, in_memory_data=(images, masks))
        
    else:
        print("Subsampled data not found. Performing subsampling and saving data...")
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
    # Rest of main function...
