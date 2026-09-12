"""
Inference for GEMS Prize: sliding window with overlap, TTA, ensemble averaging.
Produces GeoTIFF matching submission format: EPSG:32611, 100m, float32 [0,1].

Official submission requirements: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
"""

import argparse
import yaml
import numpy as np
import torch
import rasterio
from pathlib import Path
from tqdm import tqdm
from skimage.util import view_as_windows

from .dataset import load_features_and_labels, robust_normalize
from .models import get_model
from .external_data import augment_with_dem_features
from .postprocess import postprocess_pipeline

def sliding_window_inference(model, X, patch_size, overlap=0.5, batch_size=16, device="cpu", tta=True):
    """
    X: (H,W,C) normalized
    Returns: (H,W) prob map
    """
    H, W, C = X.shape
    stride = int(patch_size * (1 - overlap))
    # Pad to handle edges
    pad_h = (stride - H % stride) % stride + patch_size
    pad_w = (stride - W % stride) % stride + patch_size
    X_padded = np.pad(X, ((0, pad_h), (0, pad_w), (0,0)), mode='constant', constant_values=0)
    Hp, Wp = X_padded.shape[:2]

    # Create weight map for blending (gaussian)
    # Simple uniform averaging with count
    prob_map = np.zeros((Hp, Wp), dtype=np.float32)
    count_map = np.zeros((Hp, Wp), dtype=np.float32)

    # Prepare patches
    patches = []
    coords = []
    for i in range(0, Hp - patch_size + 1, stride):
        for j in range(0, Wp - patch_size + 1, stride):
            patch = X_padded[i:i+patch_size, j:j+patch_size, :]  # (p,p,C)
            patches.append(patch)
            coords.append((i,j))

    # Batch inference
    model.eval()
    all_probs = []
    with torch.no_grad():
        for b in range(0, len(patches), batch_size):
            batch_patches = patches[b:b+batch_size]
            batch_tensor = np.stack(batch_patches)  # (B,p,p,C)
            batch_tensor = np.moveaxis(batch_tensor, -1, 1)  # (B,C,p,p)
            batch_tensor = torch.from_numpy(batch_tensor).float().to(device)

            # TTA: 8-way
            if tta:
                # Original, hflip, vflip, rot90 etc.
                # We'll do 4 rotations + flip
                tta_probs = []
                for k in range(4):
                    # rotate
                    rot_batch = torch.rot90(batch_tensor, k, dims=[2,3])
                    logits = model(rot_batch)
                    probs = torch.sigmoid(logits)
                    # rotate back
                    probs = torch.rot90(probs, -k, dims=[2,3])
                    tta_probs.append(probs)

                    # hflip
                    flip_batch = torch.flip(rot_batch, dims=[3])
                    logits_f = model(flip_batch)
                    probs_f = torch.sigmoid(logits_f)
                    probs_f = torch.flip(probs_f, dims=[3])
                    probs_f = torch.rot90(probs_f, -k, dims=[2,3])
                    tta_probs.append(probs_f)
                # average TTA
                probs_avg = torch.stack(tta_probs).mean(dim=0)
            else:
                logits = model(batch_tensor)
                probs_avg = torch.sigmoid(logits)

            probs_np = probs_avg.cpu().numpy()  # (B,1,p,p)
            if probs_np.shape[1] == 1:
                probs_np = probs_np[:,0]  # (B,p,p)
            all_probs.extend([p for p in probs_np])

    # Blend
    for (i,j), prob in zip(coords, all_probs):
        prob_map[i:i+patch_size, j:j+patch_size] += prob
        count_map[i:i+patch_size, j:j+patch_size] += 1

    prob_map = prob_map / np.maximum(count_map, 1)
    # Crop to original
    prob_map = prob_map[:H, :W]
    return prob_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-dir", default="outputs", help="dir with .pt models")
    parser.add_argument("--out", default="submission.tif", help="output GeoTIFF")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load features
    X_orig, y_orig, feat_meta, label_meta, tags = load_features_and_labels(
        cfg["data"]["feature_path"], cfg["data"]["label_path"]
    )
    print(f"X_orig shape {X_orig.shape}")

    if cfg["data"].get("use_external_dem", False):
        X_orig = augment_with_dem_features(X_orig, resolution=cfg["metric"]["resolution_m"])
    X_norm = robust_normalize(X_orig)

    # Find models
    model_dir = Path(args.model_dir)
    model_files = list(model_dir.glob("*.pt"))
    if not model_files:
        print(f"No models found in {model_dir}, using dummy (requires training first)")
        return

    print(f"Found {len(model_files)} models for ensemble")

    # Load ensemble
    patch_size = cfg["training"]["patch_size"]
    overlap = cfg["inference"]["overlap"]
    batch_size = cfg["inference"]["batch_size"]
    tta = cfg["inference"]["tta"]

    # Read ensemble summary to know arch
    # Fallback: assume unetplusplus efficientnet-b5
    ensemble_probs = []

    for mf in model_files:
        # Parse arch from filename if possible
        # Format: model_mc0_unetplusplus_dti0.6.pt
        name = mf.stem
        arch = "unetplusplus"
        encoder = cfg["model"]["encoder"]
        # Try to extract
        parts = name.split("_")
        for p in parts:
            if p in ["unet", "unetplusplus", "deeplabv3plus", "segformer", "fpn"]:
                arch = p
        if "segformer" in arch:
            encoder = cfg["model"]["segformer_encoder"]

        in_ch = X_norm.shape[-1]
        good_ch = cfg["training"].get("good_channels")
        if good_ch is not None:
            in_ch = len(good_ch)

        print(f"Loading {mf} arch={arch} encoder={encoder} in_ch={in_ch}")
        model = get_model(arch=arch, encoder=encoder, in_channels=in_ch, classes=1, pretrained=False)
        state = torch.load(mf, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)

        prob_map = sliding_window_inference(model, X_norm, patch_size=patch_size, overlap=overlap, batch_size=batch_size, device=device, tta=tta)
        ensemble_probs.append(prob_map)

    # Average ensemble
    final_prob = np.mean(np.stack(ensemble_probs), axis=0)
    print(f"Ensemble prob shape {final_prob.shape}, min {final_prob.min()}, max {final_prob.max()}")

    # Post-process
    final_prob_pp, binary = postprocess_pipeline(final_prob, cfg["postprocess"])
    # For submission, we want probabilities, not binary, but with postprocessing enhanced
    # Use cfg threshold? Submission should be prob, not binary.
    # We'll output final_prob_pp (which includes frangi etc) as float
    # But also ensure we respect cfg inference threshold? No, submission should be prob.

    # Clip
    final_prob_pp = np.clip(final_prob_pp, 0, 1).astype(np.float32)

    # Save GeoTIFF matching sample submission
    # Use label_meta for CRS/transform, but ensure size matches original
    # If sample submission exists, use its meta
    sample_path = cfg["data"]["sample_submission_path"]
    if Path(sample_path).exists():
        with rasterio.open(sample_path) as src:
            out_meta = src.meta.copy()
            out_transform = src.transform
            out_crs = src.crs
            out_height = src.height
            out_width = src.width
    else:
        out_meta = label_meta
        out_transform = label_meta["transform"]
        out_crs = label_meta["crs"]
        out_height, out_width = y_orig.shape

    # Ensure final_prob_pp size matches
    # Crop/pad
    if final_prob_pp.shape[0] != out_height or final_prob_pp.shape[1] != out_width:
        print(f"Resizing prob map from {final_prob_pp.shape} to {(out_height, out_width)}")
        # Simple crop
        final_prob_pp = final_prob_pp[:out_height, :out_width]
        # Pad if smaller
        if final_prob_pp.shape[0] < out_height or final_prob_pp.shape[1] < out_width:
            pad_h = out_height - final_prob_pp.shape[0]
            pad_w = out_width - final_prob_pp.shape[1]
            final_prob_pp = np.pad(final_prob_pp, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    out_path = args.out
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=out_height,
        width=out_width,
        count=1,
        dtype="float32",
        crs=out_crs,
        transform=out_transform,
        nodata=None,
    ) as dst:
        dst.write(final_prob_pp, 1)

    print(f"Saved submission to {out_path}")

if __name__ == "__main__":
    main()
