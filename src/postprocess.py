"""
Post-processing for fault predictions to boost Final Round discovery.
- Low threshold for high recall (beta=0.8)
- Morphological closing to connect segments
- Frangi filter for line enhancement
- Skeletonization optional
- Length filtering

Sources:
- Mattéo et al 2021 automatic fault mapping: https://doi.org/10.1029/2020JB021269
- Hermant et al 2025 deep learning for Quaternary faults: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf
"""

import numpy as np
from skimage.morphology import skeletonize, closing, square, remove_small_objects
from skimage.filters import frangi
from scipy.ndimage import binary_dilation

def apply_threshold(pred, thresh=0.15):
    return (pred > thresh).astype(np.uint8)

def morphological_close(binary, kernel_size=3):
    return closing(binary, square(kernel_size))

def frangi_enhance(pred, scale_range=(1,10), scale_step=2):
    # pred in [0,1], frangi expects float
    # Returns vesselness
    try:
        vessel = frangi(pred, scale_range=scale_range, scale_step=scale_step, beta1=0.5, beta2=15)
        # Blend with original
        enhanced = 0.6 * pred + 0.4 * vessel
        return np.clip(enhanced, 0, 1)
    except Exception as e:
        print(f"Frangi failed: {e}, returning original")
        return pred

def connect_faults(binary, iterations=1):
    # Dilate then skeletonize to connect
    dilated = binary_dilation(binary, iterations=iterations)
    # Could skeletonize
    # skel = skeletonize(dilated)
    return dilated.astype(np.uint8)

def filter_small_faults(binary, min_length=5):
    # Remove small objects
    # min_length in pixels
    # Use remove_small_objects
    try:
        cleaned = remove_small_objects(binary.astype(bool), min_size=min_length)
        return cleaned.astype(np.uint8)
    except:
        return binary

def postprocess_pipeline(pred, config):
    """
    pred: (H,W) float [0,1]
    config: dict from config.yaml postprocess
    Returns processed float map and binary map
    """
    # Frangi
    if config.get("frangi_filter", False):
        scale_range = tuple(config.get("frangi_scale_range", [1,10]))
        pred_enh = frangi_enhance(pred, scale_range=scale_range)
    else:
        pred_enh = pred

    # Threshold
    thresh = config.get("threshold", 0.15)
    binary = apply_threshold(pred_enh, thresh=thresh)

    # Closing
    if config.get("morphological_closing", True):
        k = config.get("closing_kernel", 3)
        binary = morphological_close(binary, kernel_size=k)

    # Connect
    binary = connect_faults(binary, iterations=1)

    # Filter small
    min_len = config.get("min_fault_length", 5)
    binary = filter_small_faults(binary, min_length=min_len)

    # Optionally skeletonize for final? But submission expects probabilities, not binary.
    # So we keep float enhanced for submission, but binary for analysis.

    # For final submission, we want probabilities: use enhanced float, but boost where binary is 1
    # e.g., keep enhanced but ensure binary regions have at least thresh
    final_prob = pred_enh.copy()
    # Optionally set binary regions to max
    final_prob[binary > 0] = np.maximum(final_prob[binary > 0], thresh)

    return final_prob, binary
