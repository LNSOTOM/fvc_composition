
import os
import shutil
import numpy as np
import random
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import config_param
from affine import Affine
from dataset.calperum_dataset import CalperumDataset
from dataset.data_loaders_fold_blockcross_subsampling import (
    block_cross_validation,
    load_raw_multispectral_image,
    plot_blocks_folds
)
from torch.utils.data import Subset, DataLoader


def distribute_water(water_idxs, bins):
    """
    Distribute water_idxs proportionally across bins (list of lists).
    Modifies bins in-place.
    """
    total = sum(len(b) for b in bins)
    pool = water_idxs.copy()
    for b in bins:
        n = int(round(len(water_idxs) * len(b) / total))
        chosen = random.sample(pool, min(n, len(pool)))
        for idx in chosen:
            b.append(idx)
            pool.remove(idx)
    # distribute any remainder
    i = 0
    while pool:
        bins[i % len(bins)].append(pool.pop())
        i += 1


def rebuild_loaders(dataset, bins, water_idxs, batch_size, num_workers):
    """
    From train/val/test bins, merge train+val and stratified split:
    returns (train_loader, val_loader, test_loader)
    """
    train_bin, val_bin, test_bin = bins
    tr_val = train_bin + val_bin
    strat = [1 if idx in water_idxs else 0 for idx in tr_val]
    tr_idx, vl_idx = train_test_split(
        tr_val, test_size=0.2, random_state=42, stratify=strat
    )
    return (
        DataLoader(Subset(dataset, tr_idx), batch_size=batch_size,
                   shuffle=True, num_workers=num_workers),
        DataLoader(Subset(dataset, vl_idx), batch_size=batch_size,
                   shuffle=False, num_workers=num_workers),
        DataLoader(Subset(dataset, test_bin), batch_size=batch_size,
                   shuffle=False, num_workers=num_workers)
    )


def count_class(loader, cls):
    """
    Count tiles in loader whose mask contains class cls.
    Returns (count, total).
    """
    c = total = 0
    for _, masks in loader:
        for m in masks:
            if (m == cls).any():
                c += 1
            total += 1
    return c, total


if __name__ == '__main__':
    # PARAMETERS
    img_dir = config_param.SUBSAMPLE_IMAGE_DIR[0]
    msk_dir = config_param.SUBSAMPLE_MASK_DIR[0]
    num_blocks = 3
    batch_size = config_param.BATCH_SIZE
    num_workers = config_param.NUM_WORKERS
    # 1. PARAMETERS
    log_file   = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense/logfile.txt'
    log_dir    = os.path.dirname(log_file)

    # LOAD dataset
    imgs, msks = CalperumDataset.load_subsampled_data(img_dir, msk_dir)
    all_idxs = list(range(len(imgs)))

    # IDENTIFY water tiles
    water_idxs = [i for i, m in enumerate(msks) if (m == 4).any()]

    # BUILD combined_data
    combined = [
        (None, None,
         os.path.join(img_dir, f),
         os.path.join(msk_dir, f"mask_{f}"))
        for f in sorted(os.listdir(img_dir)) if f.lower().endswith('.tif')
    ]

    # ORIGINAL spatial CV
    dataset = CalperumDataset(img_dir, msk_dir, transform=None,
                              in_memory_data=(imgs, msks))
    folds = block_cross_validation(dataset=dataset,
                                   combined_data=combined,
                                   num_blocks=num_blocks)

    # EXTRACT split indices and remove water
    split_bins = []  # [ [fold, name, indices], ... ]
    for fi, (tr, vl, te) in enumerate(folds):
        split_bins += [
            [fi, 'train', list(tr.dataset.indices)],
            [fi, 'val',   list(vl.dataset.indices)],
            [fi, 'test',  list(te.dataset.indices)]
        ]
    for _, _, lst in split_bins:
        lst[:] = [i for i in lst if i not in water_idxs]

    # GROUP bins by fold and flatten
    grouped = {fi: [lst for f, n, lst in split_bins if f == fi]
               for fi in range(num_blocks)}
    all_bins = [lst for bins in grouped.values() for lst in bins]

    # REDISTRIBUTE water
    distribute_water(water_idxs, all_bins)

    # REBUILD loaders
    new_folds = []
    for fi in range(num_blocks):
        bins = grouped[fi]  # train, val, test lists
        new_folds.append(
            rebuild_loaders(dataset, bins, water_idxs,
                            batch_size, num_workers)
        )

    # REPORT class distributions
    for cls in [4, 1, 2]:
        print(f"\nClass-{cls} distribution after redistribution:")
        for fi, (tr, vl, te) in enumerate(new_folds):
            for name, loader in [('Train', tr), ('Val', vl), ('Test', te)]:
                c, t = count_class(loader, cls)
                print(f" Fold {fi} {name}: {c}/{t} ({c/t:.2%})")

    # PLOT final block map once
    coords = np.array([
        load_raw_multispectral_image(p)[1]['transform'] * (0, 0)
        for *_, p, _ in combined
    ])
    labels = KMeans(n_clusters=num_blocks, random_state=42)
    labels = labels.fit_predict(coords)
    fold_assign = {
        fi: {
            'train_indices': new_folds[fi][0].dataset.indices,
            'val_indices':   new_folds[fi][1].dataset.indices,
            'test_indices':  new_folds[fi][2].dataset.indices
        }
        for fi in range(num_blocks)
    }
    plot_blocks_folds(coords, labels, fold_assign, crs="EPSG:7854")

        # COPY final subset once
    out_img = os.path.join(os.path.dirname(img_dir), 'water_balanced_predictors')
    out_msk = os.path.join(os.path.dirname(msk_dir),  'water_balanced_masks')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)
    for idx in all_idxs:
        shutil.copy(combined[idx][2], os.path.join(out_img,
                     os.path.basename(combined[idx][2])))
        shutil.copy(combined[idx][3], os.path.join(out_msk,
                     os.path.basename(combined[idx][3])))

    print(f"✅ Final subset copied to {out_img} and {out_msk}")

    # SAVE balanced-water indices to JSON
    import json
    water_idx_file = os.path.join(log_dir, 'balanced_water_indices.json')
    with open(water_idx_file, 'w') as f:
        json.dump(water_idxs, f)
    print(f"Saved balanced water indices to {water_idx_file}")

    # SAVE final CV indices to JSON
    final_indices = []
    for tr, vl, te in new_folds:
        final_indices.extend(tr.dataset.indices)
        final_indices.extend(vl.dataset.indices)
        final_indices.extend(te.dataset.indices)
    final_indices = sorted(set(final_indices))
    final_idx_file = os.path.join(log_dir, 'cv_final_indices.json')
    with open(final_idx_file, 'w') as f:
        json.dump(final_indices, f)
    print(f"Saved CV final indices to {final_idx_file}")

# %%
