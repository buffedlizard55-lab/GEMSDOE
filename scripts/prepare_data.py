"""
Prepare data: check files, print meta, create sample submission if needed.
"""

import rasterio
import numpy as np
from pathlib import Path
import os

def check_data():
    # Naming drift (flagged irregularity): problem page vs reference solution vs Dropbox mirrors
    candidates = [
        "data/training_features.tif",
        "data/numeric_features.tif",
        "data/gems-geodawn-numerical-features.tif",
        "data/labels.tif",
        "data/faults.tif",
        "data/existing_faults.tif",
        "data/sample_submission.tif",
        "data/example_submission.tif",
        "data/1m_DEM_links.csv",
        "data/Digital-elevation-model-links-JSON.pdf"
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"FOUND: {p}")
            try:
                with rasterio.open(p) as src:
                    print(f"  shape: {src.width}x{src.height} count:{src.count} crs:{src.crs} res:{src.res}")
                    if src.count > 1:
                        for i in range(1, min(6, src.count+1)):
                            try:
                                print(f"    band {i}: {src.tags(i)}")
                            except:
                                pass
            except Exception as e:
                print(f"  error reading: {e}")
        else:
            print(f"MISSING: {p}")

    # If no sample submission, create dummy
    label_src = next((p for p in ["data/labels.tif", "data/faults.tif", "data/existing_faults.tif"] if os.path.exists(p)), None)
    if not os.path.exists("data/sample_submission.tif") and label_src:
        print(f"Creating dummy sample_submission.tif from {label_src}")
        with rasterio.open(label_src) as src:
            meta = src.meta.copy()
            meta.update(dtype='float32', count=1, nodata=None)
            data = np.zeros((src.height, src.width), dtype=np.float32)
            with rasterio.open("data/sample_submission.tif", "w", **meta) as dst:
                dst.write(data, 1)
        print("Created data/sample_submission.tif")

if __name__ == "__main__":
    check_data()
