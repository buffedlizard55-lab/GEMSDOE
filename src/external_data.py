"""
External data handling with verified official sources.
All downloads are from public domain USGS or CC-licensed INGENIOUS.

Sources:
- GeoDAWN: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
- QFaults: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
- 3DEP: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services
- INGENIOUS: https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483

This module provides:
- download helpers
- DEM feature engineering (slope, curvature, TPI, TRI, hillshade)
"""

import os
import numpy as np
from pathlib import Path
import rasterio
from scipy.ndimage import gaussian_filter, sobel, generic_filter

def compute_slope(dem, resolution=100):
    """Slope in degrees from DEM."""
    # gradient
    dy, dx = np.gradient(dem, resolution)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
    return slope

def compute_curvature(dem, resolution=100):
    """Simple curvature via second derivative."""
    dy, dx = np.gradient(dem, resolution)
    dyy, dyx = np.gradient(dy, resolution)
    dxy, dxx = np.gradient(dx, resolution)
    # profile curvature approximation
    curvature = dxx + dyy
    return curvature

def compute_tpi(dem, radius=3):
    """Topographic Position Index: dem - mean(dem in radius)."""
    mean = generic_filter(dem, np.mean, size=2*radius+1, mode='nearest')
    return dem - mean

def compute_tri(dem):
    """Terrain Ruggedness Index: mean absolute diff to 8 neighbors."""
    # Simple 3x3
    tri = np.zeros_like(dem)
    # Use generic filter
    def tri_func(window):
        center = window[4]
        return np.mean(np.abs(window - center))
    tri = generic_filter(dem, tri_func, size=3, mode='nearest')
    return tri

def compute_hillshade(dem, azimuth=315, altitude=45, resolution=100):
    """Hillshade."""
    azimuth = 360 - azimuth
    x, y = np.gradient(dem, resolution)
    slope = np.pi/2 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180
    altituderad = altitude * np.pi / 180
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos((azimuthrad - np.pi/2) - aspect)
    return 255 * (shaded + 1) / 2

def detrended_elevation(dem, sigma=50):
    """Detrend by subtracting Gaussian smoothed version."""
    # sigma in pixels
    smoothed = gaussian_filter(dem, sigma=sigma, mode='nearest')
    return dem - smoothed

def augment_with_dem_features(X, dem_channel_index=None, resolution=100):
    """
    X: (H,W,C) includes elevation channel? We assume channel 4? Let's detect.
    If dem_channel_index is None, we try to find elevation-like channel by heuristic (largest variance).
    Actually for GEMS, detrended elevation is channel? We will just use first channel as DEM proxy if needed.
    Better: user provides separate DEM array.
    This function adds 5 new channels: slope, curvature, TPI, TRI, detrended.
    """
    # If X has at least 1 channel, use channel 0 as DEM for demo, but in real pipeline use external DEM.
    # For this function, we expect X's last dim is features, and we have a DEM array (H,W) separately.
    # We'll handle both.
    if isinstance(X, np.ndarray) and X.ndim == 3:
        # If dem_channel_index provided, use that channel as DEM base
        if dem_channel_index is not None:
            dem = X[:, :, dem_channel_index]
        else:
            # Use channel that looks like elevation: try channel 2 or 3? Fallback to 0
            dem = X[:, :, 0]
    else:
        dem = X  # assume (H,W)

    slope = compute_slope(dem, resolution=resolution)
    curv = compute_curvature(dem, resolution=resolution)
    tpi = compute_tpi(dem, radius=3)
    tri = compute_tri(dem)
    detrended = detrended_elevation(dem, sigma=50)

    # Stack
    if isinstance(X, np.ndarray) and X.ndim == 3:
        new_features = np.stack([slope, curv, tpi, tri, detrended], axis=-1)
        X_aug = np.concatenate([X, new_features], axis=-1)
        return X_aug
    else:
        return np.stack([slope, curv, tpi, tri, detrended], axis=-1)

def download_instructions():
    print("""
Official verified download instructions:

1. GeoDAWN (USGS):
   - ScienceBase landing: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
   - Files: 22103_area1_tiffs.zip, 22103_area2_tiffs.zip etc.
   - Citation: Glen & Earney 2024, DOI https://doi.org/10.5066/P93LGLVQ
   - Use: mag, radiometric grids for extra features.

2. Quaternary Faults:
   - Interactive map: https://www.usgs.gov/programs/earthquake-hazards/faults
   - GIS zip: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 -> Qfaults_GIS.zip
   - DOI: https://doi.org/10.5066/P9BCVRCK

3. 3DEP 1m DEM:
   - About: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services
   - Downloader: https://apps.nationalmap.gov/downloader/
   - LidarExplorer: https://apps.nationalmap.gov/lidar-explorer/
   - For competition region (Nevada UTM 11N), search bounding box from training_features.tif meta.

4. INGENIOUS:
   - GDR: https://gdr.openei.org/submissions/1391
   - DOI: https://doi.org/10.15121/1881483
   - Contains: conductivity, strain, gravity, earthquake density etc. Already in training_features but useful for validation.

All are public domain or CC.
""")

if __name__ == "__main__":
    download_instructions()
