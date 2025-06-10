import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import rasterio
from rasterio.transform import Affine

import config_param
from dataset.calperum_dataset import CalperumDataset

from dask import delayed, compute
from dask.diagnostics import ProgressBar
import time

from dataset.threshold_be_subsampling import subsample_tiles, estimate_class_frequencies                                                                                          
from dataset.image_preprocessing import prep_normalise_image, load_raw_multispectral_image
from map.plot_blocks_folds import plot_blocks_folds
import json

from dataset.data_augmentation_wrapper import AlbumentationsTorchWrapper
from dataset.data_augmentation import get_train_augmentation, get_val_augmentation


def log_message(message, log_file):
    # Ensure the directory for the log file exists
    log_directory = os.path.dirname(log_file)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)
    
    # Write the log message
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def combine_and_process_paths(image_dirs, mask_dirs):
    combined_data = []

    for img_dir, mask_dir in zip(image_dirs, mask_dirs):
        region_name = extract_region_name(img_dir)
        
        # List and sort image and mask files
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

        # Create a dictionary for fast lookup of mask files
        mask_dict = {os.path.basename(mask_file): os.path.join(mask_dir, mask_file) for mask_file in mask_files}

        for index, img_file in enumerate(img_files):
            # Use the exact filenames without changing them
            mask_file = img_file.replace('tiles_multispectral', 'mask_tiles_multispectral')
            
            if mask_file in mask_dict:
                full_img_path = os.path.join(img_dir, img_file)
                full_mask_path = mask_dict[mask_file]
                combined_data.append((index, region_name, full_img_path, full_mask_path))
            else:
                raise ValueError(f"No matching mask found for image {img_file} in directory {mask_dir}")

    return combined_data

def extract_region_name(path):
    match = re.search(r'/(low|medium|dense)/', path)
    return match.group(1) if match else "unknown"

def get_site_indices(combined_data):
    site_indices = {}
    for index, region_name, _, _ in combined_data:
        if region_name not in site_indices:
            site_indices[region_name] = []
        site_indices[region_name].append(index)
    return site_indices

def save_subsampled_data(
    subsampled_images, subsampled_masks, combined_data, subsampled_indices, 
    image_subsample_dir, mask_subsample_dir, indices_save_path, combined_indices_save_path
):
    if isinstance(image_subsample_dir, str):
        image_subsample_dir = [image_subsample_dir]
    if isinstance(mask_subsample_dir, str):
        mask_subsample_dir = [mask_subsample_dir]
    
    for img_dir in image_subsample_dir:
        os.makedirs(img_dir, exist_ok=True)
    for mask_dir in mask_subsample_dir:
        os.makedirs(mask_dir, exist_ok=True)
    
    for original_idx in subsampled_indices:
        _, _, img_path, mask_path = combined_data[original_idx]
        
        img_filename = os.path.basename(img_path)
        mask_filename = os.path.basename(mask_path)
        
        img_save_path = os.path.join(image_subsample_dir[0], img_filename)
        mask_save_path = os.path.join(mask_subsample_dir[0], mask_filename)
        
        image, image_profile = load_raw_multispectral_image(img_path)
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask_profile = src.profile

        with rasterio.open(img_save_path, 'w', **image_profile) as dst:
            dst.write(image)

        with rasterio.open(mask_save_path, 'w', **mask_profile) as dst:
            dst.write(mask, 1)

    with open(indices_save_path, 'w') as f:
        json.dump(subsampled_indices, f)
    
    combined_indices = {}
    for region, indices in get_site_indices(combined_data).items():
        combined_indices[region] = indices
    with open(combined_indices_save_path, 'w') as f:
        json.dump(combined_indices, f)

    print("Subsampled images saved in the following directories:")
    for img_dir in image_subsample_dir:
        print(img_dir)
    
    print("Subsampled masks saved in the following directories:")
    for mask_dir in mask_subsample_dir:
        print(mask_dir)
    print(f"Subsampled indices saved to {indices_save_path}")
    print(f"Combined data indices saved to {combined_indices_save_path}")
    
def extract_coordinates(combined_data):
    coordinates = []
    for _, _, img_path, _ in combined_data:
        try:
            image, profile = load_raw_multispectral_image(img_path)
            transform = profile.get('transform', None)

            if transform is None:
                raise ValueError(f"Transform is missing in the profile for {img_path}.")
            
            if not isinstance(transform, Affine):
                raise TypeError(f"Transform for {img_path} is not an Affine object.")
            
            # Extract coordinates using the transform
            coords = transform * (0, 0)
            coordinates.append(coords)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if len(coordinates) == 0:
        raise ValueError("No coordinates extracted.")

    return np.array(coordinates)

def plot_with_coordinates(dataset, combined_data, indices=None, log_file_path="", crs="EPSG:7854", num_blocks=config_param.NUM_BLOCKS):
    coordinates = []

    # Extract coordinates from combined_data
    for idx, (_, _, img_path, _) in enumerate(combined_data):
        if indices is None or idx in indices:
            try:
                image, profile = load_raw_multispectral_image(img_path)
                transform = profile.get('transform', None)
                
                if transform is None:
                    raise ValueError(f"Transform is missing in the profile for {img_path}.")
                
                if not isinstance(transform, Affine):
                    raise TypeError(f"Transform for {img_path} is not an Affine object.")
                
                coords = transform * (0, 0)
                coordinates.append(coords)
            except Exception as e:
                log_message(f"Error processing {img_path}: {e}", log_file_path)

    if len(coordinates) == 0:
        log_message("No coordinates were extracted for plotting. Please check the input data.", log_file_path)
        raise ValueError("No coordinates extracted for plotting.")

    coordinates = np.array(coordinates)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)

    log_message(f"Coordinates shape after reshaping: {coordinates.shape}", log_file_path)

    if coordinates.ndim != 2:
        raise ValueError(f"Expected coordinates to be 2D after reshaping, but got shape: {coordinates.shape}")

    kmeans = KMeans(n_clusters=num_blocks, init='k-means++', random_state=42).fit(coordinates)
    block_labels = kmeans.labels_

    # Initialize fold_assignments correctly as a dictionary of lists
    fold_assignments = {block: [] for block in np.unique(block_labels)}

    for block in np.unique(block_labels):
        fold_assignments[block] = {
            'train_indices': [],  # Initialize as empty lists
            'val_indices': [],
            'test_indices': []
        }

    # Plot using the block labels obtained from KMeans
    plot_blocks_folds(coordinates, block_labels, fold_assignments, crs=crs)


def get_dataset_splits(image_folder, mask_folder, combined_data, transform, soil_threshold, soil_class=0, removal_ratio=0.5, num_classes=config_param.OUT_CHANNELS, save_augmented=False, augmented_save_dir=None, indices_save_path='subsampled_indices.json', combined_indices_save_path='combined_indices.json'):
    # log_file = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low/logfile.txt' #low
    # log_file = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium/logfile.txt' #medium
    # log_file = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense/logfile.txt' #dense
    log_file = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/logfile.txt' #dense
    start_time = time.time()

    dataset = CalperumDataset(image_folder, mask_folder, transform=None, save_augmented=False, augmented_save_dir=None)
    
    def get_data(i):
        return dataset[i][0], dataset[i][1]
    
    with ProgressBar():
        lazy_results = [delayed(get_data)(i) for i in range(len(dataset))]
        images, masks = zip(*compute(*lazy_results))
    
    log_message(f"Data extraction time: {time.time() - start_time:.2f} seconds", log_file)
    
    total_size_before_subsampling = len(images)
    log_message(f"Total dataset size before subsampling: {total_size_before_subsampling}", log_file)
    
    original_class_frequencies = estimate_class_frequencies(masks, num_classes)
    log_message(f"Original Class Frequencies: {original_class_frequencies}", log_file)
    
    log_message("Subsampling the dataset...", log_file)
    subsample_start_time = time.time()
    subsampled_images, subsampled_masks, subsampled_indices = subsample_tiles(images, masks, soil_threshold, soil_class, removal_ratio)
    log_message(f"Subsampling time: {time.time() - subsample_start_time:.2f} seconds", log_file)

    subsampled_class_frequencies = estimate_class_frequencies(subsampled_masks, num_classes)
    log_message(f"Subsampled Class Frequencies: {subsampled_class_frequencies}", log_file)

    save_subsampled_data(subsampled_images, subsampled_masks, combined_data, subsampled_indices, image_subsample_dir=config_param.SUBSAMPLE_IMAGE_DIR, mask_subsample_dir=config_param.SUBSAMPLE_MASK_DIR, indices_save_path=indices_save_path, combined_indices_save_path=combined_indices_save_path)

    log_message("Reloading the subsampled dataset for training...", log_file)
    subsampled_images, subsampled_masks = CalperumDataset.load_subsampled_data(config_param.SUBSAMPLE_IMAGE_DIR, config_param.SUBSAMPLE_MASK_DIR)
    
    dataset = CalperumDataset(image_folder, mask_folder, transform=transform, in_memory_data=(subsampled_images, subsampled_masks))

    total_size_after_subsampling = len(subsampled_images)
    log_message(f"Total dataset size after subsampling: {total_size_after_subsampling}", log_file)

    return dataset, subsampled_indices, subsampled_images, subsampled_masks 


def block_cross_validation(dataset, combined_data, num_blocks, kmeans_centroids=None):
    # log_file = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low/logfile.txt' #low
    # log_file = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium/logfile.txt' #medium
    # log_file = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense/logfile.txt' #dense
    log_file = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense/logfile.txt' #dense
    coordinates = []
    
    for idx, (_, _, img_path, _) in enumerate(combined_data):
        try:
            image, profile = load_raw_multispectral_image(img_path)
            transform = profile.get('transform', None)
            
            if transform is None:
                raise ValueError(f"Transform is missing in the profile for {img_path}.")
            
            if not isinstance(transform, Affine):
                raise TypeError(f"Transform for {img_path} is not an Affine object.")
            
            coords = transform * (0, 0)
            coordinates.append(coords)
        except Exception as e:
            log_message(f"Error processing {img_path}: {e}", log_file)

    if len(coordinates) == 0:
        log_message("No coordinates were extracted. Please check the input data.", log_file)
        raise ValueError("No coordinates extracted after subsampling.")

    coordinates = np.array(coordinates)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)

    log_message(f"Coordinates shape after reshaping: {coordinates.shape}", log_file)

    if coordinates.ndim != 2:
        raise ValueError(f"Expected coordinates to be 2D after reshaping, but got shape: {coordinates.shape}")

    # Use provided centroids for clustering, if available
    if kmeans_centroids is not None:
        kmeans = KMeans(n_clusters=num_blocks, init=kmeans_centroids, n_init=1)
    else:
        kmeans = KMeans(n_clusters=num_blocks, init='k-means++', random_state=42)

    kmeans.fit(coordinates)
    block_labels = kmeans.labels_

    data_splits = []
    fold_assignments = {}
    
    def get_dominant_class(dataset, idx):
        _, mask = dataset[idx]
        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
        mask_flat = mask_np[mask_np >= 0].flatten()  # Exclude -1
        if mask_flat.size == 0:
            return -1  # All void
        return np.bincount(mask_flat).argmax()

    for block in np.unique(block_labels):
        test_indices = [i for i in range(len(block_labels)) if block_labels[i] == block]
        train_val_indices = [i for i in range(len(block_labels)) if block_labels[i] != block]
        
        if len(train_val_indices) == 0 or len(test_indices) == 0:
            log_message(f"Skipping block {block} due to insufficient data.", log_file)
            continue

        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)
        
        # --- Albumentations transforms ---
        alb_train_transform = AlbumentationsTorchWrapper(get_train_augmentation())
        alb_val_transform = AlbumentationsTorchWrapper(get_val_augmentation())
        
        # --- Datasets with transforms ---
        train_dataset_full = CalperumDataset(transform=alb_train_transform, in_memory_data=(dataset.images, dataset.masks))
        val_dataset_full = CalperumDataset(transform=alb_val_transform, in_memory_data=(dataset.images, dataset.masks))
        test_dataset_full = CalperumDataset(transform=alb_val_transform, in_memory_data=(dataset.images, dataset.masks))

        # --- Subsets for folds ---
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # --- Class balancing REBALANCE: WeightedRandomSampler for train_loader ---
        dominant_classes = np.array([get_dominant_class(dataset, idx) for idx in train_indices])
        valid_mask = dominant_classes >= 0
        filtered_train_indices = [idx for idx, valid in zip(train_indices, valid_mask) if valid]
        dominant_classes = dominant_classes[valid_mask]
        num_classes = config_param.OUT_CHANNELS

        class_sample_counts = np.bincount(dominant_classes, minlength=num_classes)
        class_percentages = [100.0 * x / len(dominant_classes) if len(dominant_classes) > 0 else 0 for x in class_sample_counts]
        class_weights = 1. / (class_sample_counts + 1e-6)
        sample_weights = class_weights[dominant_classes]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # Print/log before rebalancing class counts
        before_msg = (
            f"\nBlock {block} - Before Weighted Sampling (train split):\n"
            f"Class counts: {class_sample_counts.tolist()}\n"
            f"Class percentages: {[f'{p:.2f}%' for p in class_percentages]}\n"
        )
        print(before_msg)
        log_message(before_msg, log_file)
        
        # Print/log after rebalancing (expected, since sampler uses inverse freq so approx uniform)
        after_counts = [int(len(dominant_classes)/num_classes)] * num_classes
        after_percentages = [f"{100.0/num_classes:.2f}%" for _ in range(num_classes)]
        after_msg = (
            f"Block {block} - Target After Weighted Sampling (train split):\n"
            f"Expected class counts per batch: ~{after_counts}\n"
            f"Expected class percentages per batch: {after_percentages}\n"
            f"Effective class distribution will be approximately uniform across classes due to sampling.\n"
        )
        print(after_msg)
        log_message(after_msg, log_file)
        
        filtered_train_dataset = Subset(dataset, filtered_train_indices)
        train_loader = DataLoader(filtered_train_dataset, batch_size=config_param.BATCH_SIZE, sampler=sampler, num_workers=config_param.NUM_WORKERS)
        # train_loader = DataLoader(train_dataset, batch_size=config_param.BATCH_SIZE, shuffle=True, num_workers=config_param.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=config_param.BATCH_SIZE, shuffle=False, num_workers=config_param.NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=config_param.BATCH_SIZE, shuffle=False, num_workers=config_param.NUM_WORKERS)

        fold_assignments[block] = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }

        data_splits.append((train_loader, val_loader, test_loader))
    
    plot_blocks_folds(coordinates, block_labels, fold_assignments, crs="EPSG:7854")

    return data_splits
