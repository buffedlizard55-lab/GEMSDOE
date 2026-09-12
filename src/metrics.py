"""
Distance-weighted Tversky Index - exact reproduction of competition metric.
Verified against problem description: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/

Formulas:
k(d) = max(1 - d/R, 0) triangular kernel, R=300m = 3 pixels at 100m
TP_w = sum_{g in G} max_{x: d(x,g)<=R} p(x) k(d(x,g))
FP_w = sum_{x:p(x)>0} p(x) [1 - max_{g in G} k(d(x,g))]
FN_w = sum_{g in G} [1 - max_{x:d(x,g)<=R} p(x) k(d(x,g))]
DTI = TP_w / (TP_w + alpha FP_w + beta FN_w + eps)

Official verified source: competition page.
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

def triangular_kernel(distance_map, R_pixels=3):
    """k(d) = max(1 - d/R, 0)"""
    return np.maximum(1.0 - distance_map / R_pixels, 0.0)

def compute_distance_weighted_tversky(pred, gt, R_pixels=3, alpha=0.2, beta=0.8, eps=1e-7, return_components=False):
    """
    pred: (H,W) float in [0,1]
    gt: (H,W) binary {0,1} or bool
    R_pixels: int, 300m / 100m = 3
    """
    pred = pred.astype(np.float32)
    gt = (gt > 0.5).astype(bool) if gt.dtype != bool else gt

    # If no GT faults, handle edge case: metric should be 1 if no pred, else penalize FP
    if not np.any(gt):
        FP_w = np.sum(pred)  # no kernel weighting because no GT
        TP_w = 0.0
        FN_w = 0.0
        dti = 0.0 if FP_w > 0 else 1.0
        if return_components:
            return dti, (TP_w, FP_w, FN_w)
        return dti

    # Distance from each pixel to nearest GT pixel
    # For FP weighting: need max_g k(d(x,g)) = k(distance_to_nearest_GT)
    # distance_transform_edt with inverted gt gives distance to nearest True
    # edt returns distance to background; so for distance to GT, we invert?
    # If gt is True for fault, we want distance to nearest True.
    # distance_transform_edt(~gt) gives distance to nearest False? Actually need check:
    # edt of binary image: distance to background (0). So if we want distance to GT (True), we do edt of ~gt.
    dist_to_gt = distance_transform_edt(~gt)  # distance to nearest GT pixel
    k_to_gt = triangular_kernel(dist_to_gt, R_pixels=R_pixels)  # max_g k(d(x,g))

    # For TP and FN: need for each GT pixel, max_{x: d(x,g)<=R} p(x) k(d(x,g))
    # This is like: for each GT pixel, look in R neighborhood for best weighted pred.
    # Efficient approximation: dilate pred weighted by kernel?
    # Brute force approach: for each GT pixel, extract window and compute max.
    # But we can use distance transform of pred-weighted? Let's implement efficient two-step:
    # Compute pred * k? No, k depends on distance between x and g.
    # For each GT pixel g, we need max over x within R of p(x) * k(dist(x,g))
    # This is equivalent to max-filter of p(x) weighted by distance to g.
    # We can approximate by: for each x, its contribution to nearby GT pixels is p(x)*k(dist). So for each GT pixel, max over neighborhood.
    # We'll implement via iterative dilation using max filter with distance-weighted pred.

    # Approach: Create an image where each pixel x has value p(x). For each GT pixel g,
    # we want max_{x in neighborhood} p(x) * (1 - d(x,g)/R)
    # We can compute via distance transform of "1 - p"? Not straightforward.
    # We'll use brute-force with uniform_filter optimization? Since R=3, window size = 2*R+1 = 7, small. So we can use view_as_windows or convolution max.
    # For each GT pixel, we need to search 7x7 window.
    from skimage.util import view_as_windows

    # Pad pred and compute kernel weights for offset
    pad = R_pixels
    pred_padded = np.pad(pred, pad, mode='constant', constant_values=0)
    H, W = gt.shape
    # Precompute distance kernel for offset
    yy, xx = np.mgrid[-R_pixels:R_pixels+1, -R_pixels:R_pixels+1]
    dist_kernel = np.sqrt(xx**2 + yy**2)
    k_kernel = triangular_kernel(dist_kernel, R_pixels=R_pixels)  # (2R+1, 2R+1)

    # For each GT pixel, find max p(x)*k
    # We can vectorize by using view_as_windows on pred_padded
    # pred_windows shape: (H, W, 2R+1, 2R+1)
    pred_windows = view_as_windows(pred_padded, (2*R_pixels+1, 2*R_pixels+1))  # (H,W,7,7) if step 1, but view_as_windows returns (H,W,7,7) after squeeze? Let's check.
    # Actually view_as_windows with window (7,7) on (H+2R, W+2R) gives (H, W, 7,7)?? Need to ensure.
    # pred_padded shape = H+2R, W+2R. Windows (7,7) -> (H, W, 7,7)
    # Then weighted
    weighted = pred_windows * k_kernel[None, None, :, :]  # broadcast
    max_weighted = np.max(weighted, axis=(2,3))  # (H,W)

    # Only consider GT pixels
    TP_w = np.sum(max_weighted[gt])
    FN_w = np.sum(1.0 - max_weighted[gt])

    # FP_w: sum_x p(x) * [1 - max_g k(d(x,g))]
    FP_w = np.sum(pred * (1.0 - k_to_gt))

    DTI = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)
    if return_components:
        return DTI, (TP_w, FP_w, FN_w)
    return DTI

def evaluate_geotiff(pred_path, true_path, R_meters=300, resolution=100, alpha=0.2, beta=0.8):
    with rasterio.open(pred_path) as src:
        pred = src.read(1).astype(np.float32)
        # handle nodata
        if src.nodata is not None:
            pred = np.where(pred == src.nodata, 0, pred)
        pred = np.clip(pred, 0, 1)
    with rasterio.open(true_path) as src:
        true = src.read(1)
        if src.nodata is not None:
            true = np.where(true == src.nodata, 0, true)
    R_pixels = int(R_meters / resolution)
    dti, (tp, fp, fn) = compute_distance_weighted_tversky(pred, true, R_pixels=R_pixels, alpha=alpha, beta=beta, return_components=True)
    print(f"DTI={dti:.5f} TP_w={tp:.2f} FP_w={fp:.2f} FN_w={fn:.2f}")
    return dti

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="predicted GeoTIFF")
    parser.add_argument("--true", required=True, help="ground truth GeoTIFF")
    parser.add_argument("--R", type=int, default=300)
    parser.add_argument("--res", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.8)
    args = parser.parse_args()
    evaluate_geotiff(args.pred, args.true, R_meters=args.R, resolution=args.res, alpha=args.alpha, beta=args.beta)
