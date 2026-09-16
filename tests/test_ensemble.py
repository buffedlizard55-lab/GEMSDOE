"""Regression tests for the parallel MC-ensemble path (2026-09-15).

Covers the three pieces that only exist for the runner-scale ensemble:
  1. src.train._compact_window_subset - the calibration crop must be compact and
     fault-bearing, otherwise the shaping search scores (and pays EDT for) the
     whole raster.
  2. src.inference --raw flag parsing/wiring (no postprocess, no shaping).
  3. scripts/blend_submission.py end-to-end on two synthetic folds: grid, range,
     NaN footprint, report internals and the metric closed-form identity.

Fast: numpy/rasterio only except the inference import (torch).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- 1
def test_compact_window_subset_picks_contiguous_faulty_run():
    from src.train import _compact_window_subset

    patch = 16
    tw = [(0, j * patch) for j in range(10)]                      # one row, 10 windows
    gt = np.zeros((10, patch, patch), np.float32)
    gt[0, 8, 8] = 1.0                                             # only window 0 has fault
    sel = _compact_window_subset(tw, gt, nmax=3, patch=patch)
    assert len(sel) == 3
    assert (0, 0) in sel                                          # the fault-bearing window is in
    rows = {i for i, _ in sel}
    cols = sorted(j for _, j in sel)
    assert rows == {0}                                            # bbox stays on one row -> compact
    assert cols[-1] - cols[0] <= 2 * patch                          # contiguous, not scattered

    # no fault anywhere -> deterministic fallback to the first nmax
    sel2 = _compact_window_subset(tw, np.zeros_like(gt), nmax=3, patch=patch)
    assert sel2 == tw[:3]


# --------------------------------------------------------------------------- 2
def test_inference_raw_flag_exists():
    src = (ROOT / "src" / "inference.py").read_text()
    assert '"--raw"' in src
    assert "args.raw" in src
    # raw must skip BOTH post-processing and shaping: the `if args.raw:` branch
    # contains no postprocess call, and shaping is gated by `not args.raw`.
    raw_branch = src.split('if args.raw:', 1)[1].split("else:", 1)[0]
    assert "postprocess_pipeline" not in raw_branch
    assert 'if not args.raw and (shp.get("t0")' in src


# --------------------------------------------------------------------------- 3
def _write_tif(path, arr, crs="EPSG:32611"):
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype="float32", crs=crs, transform=rasterio.Affine(100, 0, 0, 0, -100, 0),
                   compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def test_blend_end_to_end(tmp_path):
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):                       # diagonal fault
        gt[k, k] = 1.0
    rng = np.random.default_rng(0)

    folds = []
    for fi in range(2):
        fd = tmp_path / f"fold-{fi}"
        fd.mkdir()
        # a "model" that fires near the true fault with noise
        from scipy.ndimage import gaussian_filter, binary_dilation
        band = gaussian_filter(binary_dilation(gt > 0.5, iterations=1).astype(np.float32), 1.4)
        prob = np.clip(0.02 + 0.95 * band + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)
        prob[:4, :4] = np.nan                    # fake footprint hole
        _write_tif(fd / "prob_raw.tif", prob)
        np.savez_compressed(fd / f"heldout_mc{fi}.npz",
                            pred=prob[8:48, 8:48].astype(np.float16),
                            gt=gt[8:48, 8:48].astype(np.uint8))
        (fd / "manifest.json").write_text(json.dumps({"models": [{"file": f"m{fi}.pt", "dti": 0.5}]}))
        folds.append(str(fd))

    _write_tif(tmp_path / "sample.tif", np.zeros((H, W), np.float32))
    _write_tif(tmp_path / "labels.tif", gt)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""
training: {alpha: 0.2, beta: 0.8}
metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, epsilon: 1.0e-7}
""")
    out = tmp_path / "submission.tif"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", *folds, "--config", str(cfg),
         "--sample", str(tmp_path / "sample.tif"), "--labels", str(tmp_path / "labels.tif"),
         "--out", str(out), "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr

    with rasterio.open(out) as src:
        q = src.read(1)
        assert q.shape == (H, W)
        assert src.count == 1 and src.dtypes[0] == "float32" and src.crs.to_epsg() == 32611
    assert np.isnan(q[:4, :4]).all(), "NaN footprint must be preserved through the blend"
    fin = q[np.isfinite(q)]
    assert (fin >= 0).all() and (fin <= 1).all()
    assert fin.max() == 1.0 and (fin > 0).sum() > 0            # shaped hard map fires somewhere

    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["n_folds"] == 2
    assert rep["submission"]["finite_px"] == int(np.isfinite(q).sum())
    assert len(rep["shaping"]["calibration_table"]) == 1 + 2 * 11  # row0 + grid x thin options
    loc = rep["local_score_vs_known"]
    assert loc["closed_form_matches"] is True, "DTI != closed form -> identity regression"
    assert 0.0 <= loc["dti_known_faults"] <= 1.0
    # the blended+shaped map should beat the all-zeros baseline on this toy gt
    assert loc["TP_w"] > 0.0


def test_blend_refuses_all_zero_submission(tmp_path):
    """Guard added 2026-09-15: if shaping collapses the whole map (a failure mode seen
    when a pre-transform flattens the distribution), the blend must FAIL LOUDLY,
    never write an empty submission."""
    H = W = 64
    gt = np.zeros((H, W), np.float32)
    for k in range(8, 56):
        gt[k, k] = 1.0
    fd = tmp_path / "fold-flat"
    fd.mkdir()
    flat = np.full((H, W), 0.01, np.float32)          # every pixel below every floor
    _write_tif(fd / "prob_raw.tif", flat)
    np.savez_compressed(fd / "heldout_mc0.npz", pred=flat.astype(np.float16),
                        gt=gt.astype(np.uint8))
    (fd / "manifest.json").write_text(json.dumps({"models": [{"file": "m.pt", "dti": 0.01}]}))
    _write_tif(tmp_path / "labels.tif", gt)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""
training: {alpha: 0.2, beta: 0.8}
metric: {R_meters: 300, resolution_m: 100, alpha: 0.2, beta: 0.8, epsilon: 1.0e-7}
""")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "blend_submission.py"),
         "--folds", str(fd), "--config", str(cfg), "--labels", str(tmp_path / "labels.tif"),
         "--out", str(tmp_path / "sub.tif"), "--report", str(tmp_path / "rep.json")],
        capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "collapsed" in (r.stderr + r.stdout), "all-zero submission must abort the blend"
