"""
Training loop for GEMS Prize - top leaderboard approach.
- Monte Carlo CV splits
- Multiple architectures ensemble
- Tversky + Focal + BCE loss
- Distance-weighted Tversky for validation
- Early stopping, mixed precision, scheduler

Official competition: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
Metric: distance-weighted Tversky alpha=0.2 beta=0.8 R=300m
"""

import argparse
import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from pathlib import Path

from .dataset import load_features_and_labels, make_patches, robust_normalize, FaultDataset
from .models import get_model, count_params
from .losses import CombinedLoss, TverskyLoss
from .metrics import compute_distance_weighted_tversky
from .external_data import augment_with_dem_features

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="train"):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device, R_pixels=3, alpha=0.2, beta=0.8):
    model.eval()
    all_preds = []
    all_trues = []
    # For metric, we need to reconstruct? For val patches, we compute per patch DTI then average
    # Simpler: compute DTI per patch and average
    dtis = []
    for X, y in tqdm(loader, desc="val"):
        X = X.to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_np = y.cpu().numpy()
        # probs shape (B,1,H,W) -> (B,H,W)
        if probs.ndim == 4:
            probs = probs[:,0]
        for i in range(probs.shape[0]):
            dti = compute_distance_weighted_tversky(probs[i], y_np[i], R_pixels=R_pixels, alpha=alpha, beta=beta)
            dtis.append(dti)
    mean_dti = np.mean(dtis) if dtis else 0
    return mean_dti

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X_orig, y_orig, feat_meta, label_meta, tags = load_features_and_labels(
        cfg["data"]["feature_path"], cfg["data"]["label_path"]
    )
    print(f"Original X shape: {X_orig.shape}, y shape: {y_orig.shape}")

    # Optional DEM augmentation
    if cfg["data"].get("use_external_dem", False):
        print("Augmenting with DEM features (slope, curvature, TPI, TRI, detrended)")
        X_orig = augment_with_dem_features(X_orig, resolution=cfg["metric"]["resolution_m"])
        print(f"Augmented X shape: {X_orig.shape}")

    # Normalize
    X_norm = robust_normalize(X_orig)
    print(f"Normalized X shape: {X_norm.shape}")

    n_features = X_norm.shape[-1]
    cfg["model"]["in_channels"] = n_features

    # Training params
    patch_size = cfg["training"]["patch_size"]
    test_prop = cfg["training"]["test_proportion"]
    mc_splits = cfg["training"]["mc_splits"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = float(cfg["training"]["init_lr"])  # coerce: YAML parses 1e-4 as str (bug found in E2E test)
    alpha = cfg["training"]["alpha"]
    beta = cfg["training"]["beta"]
    R_pixels = cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"]

    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_models = []

    for mc in range(mc_splits):
        print(f"\n=== MC Split {mc+1}/{mc_splits} ===")
        X_train, y_train, X_test, y_test, test_inds = make_patches(
            X_norm, y_orig, patch_size, test_proportion=test_prop, seed=mc*10, train_step=cfg["training"]["train_step"]
        )
        print(f"Train patches: {X_train.shape}, Test patches: {X_test.shape}")

        # Filter good channels if specified
        good_ch = cfg["training"].get("good_channels")
        if good_ch is not None:
            X_train = X_train[:, good_ch, :, :]
            X_test = X_test[:, good_ch, :, :]
            in_ch = len(good_ch)
        else:
            in_ch = n_features

        train_ds = FaultDataset(X_train, y_train)
        test_ds = FaultDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg["training"].get("num_workers",0))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=cfg["training"].get("num_workers",0))

        # Model - cycle through architectures for ensemble diversity
        arch_list = cfg["model"]["architectures"]
        arch = arch_list[mc % len(arch_list)]
        encoder = cfg["model"]["encoder"] if "segformer" not in arch else cfg["model"]["segformer_encoder"]
        print(f"Training arch={arch} encoder={encoder} in_ch={in_ch}")

        model = get_model(arch=arch, encoder=encoder, in_channels=in_ch, classes=1, pretrained=cfg["model"]["pretrained"])
        print(f"Params: {count_params(model)}")
        model = model.to(device)

        criterion = CombinedLoss(alpha=alpha, beta=beta)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["training"]["weight_decay"]))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        best_dti = -1
        best_state = None
        patience = cfg["training"]["early_stopping_patience"]
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_dti = validate(model, test_loader, device, R_pixels=R_pixels, alpha=alpha, beta=beta)
            print(f"Train loss: {train_loss:.4f} Val DTI: {val_dti:.4f}")
            scheduler.step(val_dti)

            if val_dti > best_dti:
                best_dti = val_dti
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"New best DTI: {best_dti:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Save best model for this split
        model_path = output_dir / f"model_mc{mc}_{arch}_dti{best_dti:.4f}.pt"
        torch.save(best_state, model_path)
        print(f"Saved {model_path}")

        # For final ensemble, we need to keep model architecture info
        best_models.append((str(model_path), arch, encoder, best_dti))

    # Save ensemble summary
    with open(output_dir / "ensemble_summary.txt", "w") as f:
        for path, arch, enc, dti in best_models:
            f.write(f"{path} {arch} {enc} {dti}\n")
    print("Training complete. Ensemble summary saved.")

if __name__ == "__main__":
    main()
