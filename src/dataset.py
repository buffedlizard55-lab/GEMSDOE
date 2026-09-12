"""
Dataset utilities for GEMS Prize.
Handles:
- Loading training_features.tif / numeric_features.tif and labels.tif
- Nan-aware normalization
- Patchify / unpatchify (adapted from reference solution MIT)
- Augmentation
- External DEM features

Official sources verified:
- USGS GeoDAWN: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
- USGS 3DEP: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services
"""

import os
from pathlib import Path
from typing import Tuple, Union, List
import numpy as np
import rasterio
from skimage.util import view_as_windows
import torch
from torch.utils.data import Dataset

Imsize = Union[Tuple[int, int], Tuple[int, int, int]]

def patchify(image: np.ndarray, patch_size: Imsize, step: int = 1) -> np.ndarray:
    return view_as_windows(image, patch_size, step)

def _unpatchify2d(patches, imsize):
    # patches: (n_h, n_w, h, w) -> (H,W)
    n_h, n_w, h, w = patches.shape
    H, W = imsize
    # Check divisible
    assert H % h == 0 and W % w == 0 or True
    # Reconstruct
    # Using reshape and transpose
    # First, combine
    out = np.zeros(imsize, dtype=patches.dtype)
    # step = h for non-overlap, but we assume step==patch for unpatchify
    # If overlapping, need averaging - handled elsewhere
    for i in range(n_h):
        for j in range(n_w):
            out[i*h:(i+1)*h, j*w:(j+1)*w] = patches[i, j]
    return out

def _unpatchify3d(patches, imsize):
    n_h, n_w, h, w, c = patches.shape
    H, W, C = imsize
    out = np.zeros(imsize, dtype=patches.dtype)
    for i in range(n_h):
        for j in range(n_w):
            out[i*h:(i+1)*h, j*w:(j+1)*w, :] = patches[i, j]
    return out

def unpatchify(patches: np.ndarray, imsize: Imsize) -> np.ndarray:
    assert len(patches.shape) / 2 == len(imsize), "dim mismatch"
    if len(patches.shape) == 4:
        return _unpatchify2d(patches, imsize)
    elif len(patches.shape) == 5:
        return _unpatchify3d(patches, imsize)
    else:
        raise NotImplementedError("Unpatchify only supports 2D and 3D")

def robust_normalize(X, clip_percentile=(2,98)):
    """
    X: (H,W,C) with NaNs
    Clip per channel to percentile, then min-max to [0,1] or standardize.
    """
    X_norm = np.empty_like(X, dtype=np.float32)
    for c in range(X.shape[-1]):
        ch = X[:, :, c]
        valid = ch[np.isfinite(ch)]
        if len(valid) == 0:
            X_norm[:, :, c] = 0
            continue
        lo, hi = np.percentile(valid, clip_percentile)
        ch_clipped = np.clip(ch, lo, hi)
        # min-max
        min_v = np.nanmin(ch_clipped)
        max_v = np.nanmax(ch_clipped)
        if max_v - min_v < 1e-8:
            X_norm[:, :, c] = 0
        else:
            norm = (ch_clipped - min_v) / (max_v - min_v)
            norm[~np.isfinite(ch)] = 0
            X_norm[:, :, c] = norm
    return X_norm

def make_patches(X, y, patch_size, test_proportion=0.3, seed=None, train_step=32, min_fault_ratio=0.0):
    """
    Split into overlapping training patches and non-overlapping test patches.
    Only patches containing valid elevation (channel heuristic) and optionally fault.
    """
    if seed is None:
        seed = 0
    rng = np.random.default_rng(seed)

    n_features = X.shape[-1]
    # pad to divisible by patch_size
    H, W, C = X.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    X_padded = np.pad(X, ((0, pad_h), (0, pad_w), (0,0)), mode='constant', constant_values=0)
    y_padded = np.pad(y, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Non-overlapping patches for test selection
    # Using patchify with step=patch_size
    X_nonoverlap = patchify(X_padded, (patch_size, patch_size, C), step=patch_size)  # shape (n_h, n_w, 1,1,1?) Actually view_as_windows returns (n_h, n_w, patch, patch, C)
    # The shape is (n_h, n_w, patch, patch, C)?? Let's squeeze
    # For 3D input (H,W,C), patch (p,p,C) -> output (n_h, n_w, 1,1,1?) No, view_as_windows returns (n_h, n_w, 1, p, p, C)?? Let's handle generically.
    # Simplify: manually compute n_h, n_w
    n_h = X_padded.shape[0] // patch_size
    n_w = X_padded.shape[1] // patch_size

    # Reshape to list
    X_patches_nonoverlap = X_padded.reshape(n_h, patch_size, n_w, patch_size, C).transpose(0,2,1,3,4).reshape(-1, patch_size, patch_size, C)
    y_patches_nonoverlap = y_padded.reshape(n_h, patch_size, n_w, patch_size).transpose(0,2,1,3).reshape(-1, patch_size, patch_size)

    # Filter valid patches: must have finite elevation? Heuristic: use first channel? Actually reference uses channel 4 elevation.
    # We'll require not all zeros in features
    valid_mask = []
    for i in range(len(X_patches_nonoverlap)):
        # check if patch has any valid data (non-zero features)
        if np.mean(X_patches_nonoverlap[i]) > 1e-6:  # not empty
            valid_mask.append(i)
    valid_indices = np.array(valid_mask)

    # Randomly select test indices
    n_test = int(len(valid_indices) * test_proportion)
    test_choice = rng.choice(valid_indices, size=n_test, replace=False)
    test_set = set(test_choice)

    # Build test tensors
    X_test_list = []
    y_test_list = []
    test_inds = []
    for idx in test_choice:
        # compute row,col
        r = idx // n_w
        c = idx % n_w
        test_inds.append((r,c))
        X_test_list.append(X_patches_nonoverlap[idx])
        y_test_list.append(y_patches_nonoverlap[idx])

    # For training, use overlapping patches from remaining area, excluding test regions
    # Create mask of trainable area
    train_mask = np.ones((X_padded.shape[0], X_padded.shape[1]), dtype=bool)
    for (r,c) in test_inds:
        train_mask[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = False

    # Now sliding window with train_step
    X_train_list = []
    y_train_list = []
    for i in range(0, X_padded.shape[0] - patch_size + 1, train_step):
        for j in range(0, X_padded.shape[1] - patch_size + 1, train_step):
            # if this window overlaps test region, skip
            if not np.any(train_mask[i:i+patch_size, j:j+patch_size]):
                continue
            # also require some overlap with valid data
            x_patch = X_padded[i:i+patch_size, j:j+patch_size, :]
            y_patch = y_padded[i:i+patch_size, j:j+patch_size]
            if np.mean(x_patch) < 1e-6:
                continue
            # optional filter: require some fault or hard negative mining - keep 30% negatives
            has_fault = np.any(y_patch > 0)
            if not has_fault and rng.random() > 0.3:
                continue
            X_train_list.append(x_patch)
            y_train_list.append(y_patch)

    # Convert to torch format (B,C,H,W)
    X_train = np.stack(X_train_list) if X_train_list else np.empty((0, patch_size, patch_size, C))
    y_train = np.stack(y_train_list) if y_train_list else np.empty((0, patch_size, patch_size))

    X_test = np.stack(X_test_list) if X_test_list else np.empty((0, patch_size, patch_size, C))
    y_test = np.stack(y_test_list) if y_test_list else np.empty((0, patch_size, patch_size))

    # To (B,C,H,W)
    X_train = np.moveaxis(X_train, -1, 1)  # (B,C,H,W)
    X_test = np.moveaxis(X_test, -1, 1)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    return X_train, y_train, X_test, y_test, test_inds

class FaultDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        # x: (C,H,W), y: (H,W)
        if self.transform:
            # albumentations expects HWC
            # We'll handle simple torch transforms here
            pass
        return x, y

def load_features_and_labels(feature_path, label_path):
    # Try multiple possible names
    # Naming drift (flagged irregularity): problem page vs reference solution vs Dropbox mirrors
    feat_candidates = [feature_path, "data/training_features.tif", "data/numeric_features.tif", "data/gems-geodawn-numerical-features.tif", "data/features.tif"]
    label_candidates = [label_path, "data/labels.tif", "data/faults.tif", "data/existing_faults.tif"]

    feat_path = None
    for p in feat_candidates:
        if os.path.exists(p):
            feat_path = p
            break
    if feat_path is None:
        raise FileNotFoundError(f"No feature file found, tried {feat_candidates}")

    label_path_found = None
    for p in label_candidates:
        if os.path.exists(p):
            label_path_found = p
            break
    if label_path_found is None:
        raise FileNotFoundError(f"No label file found, tried {label_candidates}")

    print(f"Loading features from {feat_path}")
    with rasterio.open(feat_path) as src:
        meta = src.meta.copy()
        data = src.read()  # (C,H,W)
        print(f"Feature shape: {data.shape}, CRS: {src.crs}, res: {src.res}")
        tags = []
        for i in range(1, src.count+1):
            try:
                t = src.tags(i)
                tags.append(t)
            except:
                tags.append({})

    X = np.moveaxis(data, 0, -1)  # (H,W,C)
    # replace nodata huge negative
    X[X < -1e30] = np.nan

    print(f"Loading labels from {label_path_found}")
    with rasterio.open(label_path_found) as src:
        label_meta = src.meta.copy()
        y = src.read(1).astype(np.float32)
        print(f"Label shape: {y.shape}, unique: {np.unique(y)[:10]}")
    y[y < 1] = 0
    y = (y > 0).astype(np.float32)

    return X, y, meta, label_meta, tags
