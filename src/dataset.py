"""Dataset utilities for the GEMS Prize.

Responsibilities
----------------
* load the competition GeoTIFF stack(s) (all documented name variants are accepted; see
  data/README.md - the problem page, the reference solution and the Dropbox mirrors each
  use a different file name)
* normalisation with *persisted* statistics (train and inference must use the same numbers;
  Official Rules 3.5 requires assets that "sufficiently reproduce the winning results")
* patching that follows the reference solution's anti-leakage design: test windows are cut
  out of the global raster FIRST, zeroed globally, and only then are (overlapping) training
  windows sampled - so no training window can read a test label
* augmentation (flips / 90-degree rotations / noise) applied identically to x, y and the
  false-positive weight map
* per-patch precomputation of the metric's FP weight map (1 - max_g k(d(x,g))), cropped from
  the GLOBAL distance transform so predictions at a patch border are not over-penalised

Verified sources
----------------
- reference solution (patchify/unpatchify + global zeroing idea):
  https://github.com/drivendataorg/gems-prize-reference-solution  (cells 9-12)
- feature/label description + CRS/resolution:
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#datasets
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from torch.utils.data import Dataset

FEATURE_NAME_CANDIDATES = (
    "training_features.tif",              # problem page
    "numeric_features.tif",               # reference solution
    "gems-geodawn-numerical-features.tif",  # Dropbox mirror on the data tab
    "features.tif",
)
LABEL_NAME_CANDIDATES = ("labels.tif", "existing_faults.tif", "faults.tif")   # page / mirror / -
SAMPLE_NAME_CANDIDATES = ("sample_submission.tif", "example_submission.tif")


def resolve_path(explicit: Optional[str], candidates: Sequence[str], subdirs=("", "reconstructed")) -> str:
    """Return first existing path among `explicit` then `<dir>/<candidate>` for dir in subdirs."""
    if explicit:
        if os.path.exists(explicit):
            return explicit
        base = Path(explicit).name
        for d in subdirs:
            p = Path("data") / d / base if d else Path("data") / base
            if p.exists():
                return str(p)
    for d in subdirs:
        for c in candidates:
            p = (Path("data") / d / c) if d else (Path("data") / c)
            if p.exists():
                return str(p)
    # reconstructed variants
    for d in subdirs:
        for c in candidates:
            p = (Path("data") / d / f"recon_{c}") if d else None
            if p and p.exists():
                return str(p)
    raise FileNotFoundError(
        f"none of {list(candidates)} found under data/ (checked subdirs {list(subdirs)}). "
        "Run: bash scripts/download_competition_data.sh"
    )


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_stack(path: str) -> Tuple[np.ndarray, dict, List[dict]]:
    """-> (H,W,C) float32 array with nodata->NaN, meta dict, per-band tag dicts."""
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        meta["crs"] = src.crs
        meta["transform"] = src.transform
        data = src.read().astype(np.float32)          # (C,H,W)
        nodata = src.nodata
        tags = []
        for i in range(1, src.count + 1):
            t = dict(src.tags(i) or {})
            tags.append(t)
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    data[data < -1e30] = np.nan                        # reference solution convention
    return np.moveaxis(data, 0, -1), meta, tags         # (H,W,C)


def load_labels(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        meta["crs"] = src.crs
        meta["transform"] = src.transform
        y = src.read(1).astype(np.float32)
        if src.nodata is not None:
            y[y == src.nodata] = 0
    y[~np.isfinite(y)] = 0
    y = (y > 0.5).astype(np.float32)          # reference: `y_orig[y_orig < 1] = 0`
    return y, meta


def band_names(tags: List[dict], n: int) -> List[str]:
    out = []
    for i in range(n):
        t = tags[i] if i < len(tags) else {}
        out.append(t.get("description") or t.get("name") or t.get("LONG_NAME") or f"band_{i}")
    return out


# --------------------------------------------------------------------------------------
# normalisation (stats persisted so train == inference == reproduction)
# --------------------------------------------------------------------------------------
def fit_norm_stats(X: np.ndarray, clip_percentile=(1.0, 99.0)) -> dict:
    """Per-channel (lo, hi, mean, std) on finite values, after percentile clipping."""
    stats = {"clip_percentile": list(clip_percentile), "channels": []}
    for c in range(X.shape[-1]):
        ch = X[..., c]
        v = ch[np.isfinite(ch)]
        if v.size == 0:
            stats["channels"].append(dict(lo=0.0, hi=1.0, mean=0.0, std=1.0, n=0))
            continue
        lo, hi = (float(np.percentile(v, clip_percentile[0])), float(np.percentile(v, clip_percentile[1])))
        if not hi > lo:
            lo, hi = float(v.min()), float(v.max())
        sel = v[(v >= lo) & (v <= hi)]
        mean = float(sel.mean()) if sel.size else 0.0
        std = float(sel.std()) if sel.size and sel.std() > 1e-8 else 1.0
        stats["channels"].append(dict(lo=lo, hi=hi, mean=mean, std=std, n=int(v.size)))
    return stats


def apply_norm_stats(X: np.ndarray, stats: dict, mode: str = "clip_zscore") -> np.ndarray:
    """(H,W,C) -> (H,W,C) float32 in ~[0,1]; NaNs become 0 (model sees 'no data')."""
    out = np.zeros(X.shape, dtype=np.float32)
    for c, ch in enumerate(stats["channels"]):
        v = X[..., c].astype(np.float32)
        lo, hi = ch["lo"], ch["hi"]
        if hi <= lo:
            continue
        v = np.clip(v, lo, hi)
        if mode == "minmax":
            out[..., c] = (v - lo) / (hi - lo)
        else:  # clip_zscore -> 0..1, centred, robust to outliers
            out[..., c] = np.clip(0.5 + 0.25 * (v - ch["mean"]) / ch["std"], 0.0, 1.0)
        bad = ~np.isfinite(X[..., c])
        if bad.any():
            out[..., c][bad] = 0.0
    return out


def save_norm_stats(path, stats):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)


def load_norm_stats(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------------------
def _windows(shape_hw, patch: int, step: int):
    H, W = shape_hw
    ys = list(range(0, max(1, H - patch + 1), step))
    xs = list(range(0, max(1, W - patch + 1), step))
    if ys and ys[-1] + patch < H:
        ys.append(H - patch)
    if xs and xs[-1] + patch < W:
        xs.append(W - patch)
    return [(y, x) for y in ys for x in xs]


def make_patches(
    X: np.ndarray,
    y: np.ndarray,
    patch_size: int = 128,
    train_step: int = 64,
    test_proportion: float = 0.3,
    seed: int = 0,
    neg_fraction: float = 0.35,
    R_pixels: int = 3,
    min_px_per_patch: int = 3,
):
    """Reference-style leakage-free split + overlapping training windows.

    Returns dict with:
      X_tr, y_tr, fpw_tr (lists of arrays), X_te, y_te, test_origin (row, col of each test
      window in the padded global grid), n_gt.

    Order of operations (matches reference notebook cells 11-12 semantics):
      1. non-overlapping test grid -> choose test windows -> ZERO them in the global arrays
      2. re-patchify the *zeroed* global arrays with `train_step` overlap
      3. keep windows that contain >= min_px_per_patch fault pixels, plus `neg_fraction`
         of empty windows (hard negatives make the model calibrate, unlike the reference,
         which trains only on fault-containing windows)
    """
    rng = np.random.default_rng(seed)
    H, W, C = X.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    Xp = np.pad(X, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    yp = np.pad(y, ((0, pad_h), (0, pad_w)), constant_values=0)
    Hp, Wp = yp.shape

    # ---- 1. test windows on a non-overlapping grid --------------------------------
    grid = _windows((Hp, Wp), patch_size, patch_size)
    valid = [(i, j) for (i, j) in grid if np.isfinite(Xp[i:i + patch_size, j:j + patch_size]).any()]
    n_test = int(round(test_proportion * len(valid)))
    test_windows = sorted(rng.choice(len(valid), size=n_test, replace=False).tolist())
    test_windows = [valid[i] for i in test_windows]
    test_mask = np.zeros((Hp, Wp), bool)
    for (i, j) in test_windows:
        test_mask[i:i + patch_size, j:j + patch_size] = True

    X_test = np.stack([Xp[i:i + patch_size, j:j + patch_size] for (i, j) in test_windows]) if test_windows \
        else np.zeros((0, patch_size, patch_size, C), np.float32)
    y_test = np.stack([yp[i:i + patch_size, j:j + patch_size] for (i, j) in test_windows]) if test_windows \
        else np.zeros((0, patch_size, patch_size), np.float32)

    # ---- 2. zero the test region globally, THEN extract overlapping train windows --
    Xtr_src = Xp.copy()
    ytr_src = yp.copy()
    fp_src = (yp > 0.5)
    # global FP weight map = 1 - max_g k(d(x,g)), computed on the TRAINING labels only
    from scipy.ndimage import distance_transform_edt
    if fp_src.any():
        d2gt = distance_transform_edt(~fp_src)
        fpw_global = 1.0 - np.maximum(1.0 - d2gt / float(R_pixels), 0.0)
    else:
        fpw_global = np.ones((Hp, Wp), np.float32)
    Xtr_src[test_mask] = 0.0
    ytr_src[test_mask] = 0.0

    cand = _windows((Hp, Wp), patch_size, train_step)
    pos, neg = [], []
    for (i, j) in cand:
        if test_mask[i:i + patch_size, j:j + patch_size].mean() > 0.25:
            continue                       # mostly-test window: skip outright
        n_fault = int((ytr_src[i:i + patch_size, j:j + patch_size] > 0.5).sum())
        (pos if n_fault >= min_px_per_patch else neg).append((i, j))
    keep = list(pos)
    if neg and neg_fraction > 0:
        n_keep = int(round(neg_fraction * len(pos) / max(1e-6, 1 - neg_fraction)))
        n_keep = min(n_keep, len(neg))
        keep += [neg[a] for a in rng.choice(len(neg), size=n_keep, replace=False)]
    keep = sorted(keep)

    X_train = np.stack([Xtr_src[i:i + patch_size, j:j + patch_size] for (i, j) in keep]) if keep \
        else np.zeros((0, patch_size, patch_size, C), np.float32)
    y_train = np.stack([ytr_src[i:i + patch_size, j:j + patch_size] for (i, j) in keep]) if keep \
        else np.zeros((0, patch_size, patch_size), np.float32)
    fpw_train = np.stack([fpw_global[i:i + patch_size, j:j + patch_size] for (i, j) in keep]) if keep \
        else np.zeros((0, patch_size, patch_size), np.float32)

    summary = dict(
        n_train=len(keep), n_test=len(test_windows), n_pos=len(pos), n_neg_kept=len(keep) - len(pos),
        patch=patch_size, train_step=train_step, seed=int(seed), H=Hp, W=Wp, C=C,
        test_windows=[(int(i), int(j)) for (i, j) in test_windows],
        train_windows=[(int(i), int(j)) for (i, j) in keep],
    )
    return dict(X_train=X_train, y_train=y_train, fpw_train=fpw_train,
                X_test=X_test, y_test=y_test, summary=summary)


# --------------------------------------------------------------------------------------
# torch dataset
# --------------------------------------------------------------------------------------
class FaultDataset(Dataset):
    """X: (N,H,W,C) normalised, y: (N,H,W), fpw: (N,H,W).  Augment = flips/rot90/noise.

    Augmentation is geometry-preserving-with-label (90-degree rotations and flips keep
    faults valid; the reference solution uses RandomResizedCrop + RandomRotation(30), which
    we additionally support via `rand_crop_scale`).
    """

    def __init__(self, X, y, fpw=None, train=False, augment=True, noise_std=0.01,
                 rand_crop_scale=(0.75, 1.0), seed=None):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.fpw = np.ones_like(self.y, np.float32) if fpw is None else np.ascontiguousarray(fpw, np.float32)
        self.train = train
        self.augment = augment and train
        self.noise_std = noise_std
        self.rand_crop_scale = rand_crop_scale
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, y, w = self.X[idx], self.y[idx], self.fpw[idx]
        if self.augment:
            r = self._rng
            # random-resized-crop (zoom into a sub-window, then re-pad to patch size):
            # teaches scale invariance; faults are 1-px wide at 100 m so we keep s >= 0.75
            s_lo, s_hi = self.rand_crop_scale
            if r.random() < 0.5 and s_hi < 1.0 + 1e-6 and s_lo < 1.0:
                s = float(r.uniform(s_lo, s_hi))
                p = x.shape[0]
                cp = max(16, int(round(p * s)))
                if cp < p:
                    i = int(r.integers(0, p - cp + 1))
                    j = int(r.integers(0, p - cp + 1))
                    x, y, w = x[i:i + cp, j:j + cp], y[i:i + cp, j:j + cp], w[i:i + cp, j:j + cp]
                    k = int(round((p - cp) / 2))
                    x = np.pad(x, ((k, p - cp - k), (k, p - cp - k), (0, 0)))
                    y = np.pad(y, ((k, p - cp - k), (k, p - cp - k)))
                    w = np.pad(w, ((k, p - cp - k), (k, p - cp - k)), constant_values=1.0)
            if r.random() < 0.5:
                x, y, w = x[:, ::-1], y[:, ::-1], w[:, ::-1]
            if r.random() < 0.5:
                x, y, w = x[::-1], y[::-1], w[::-1]
            k = int(r.integers(0, 4))
            if k:
                x, y, w = (np.rot90(a, k, axes=(0, 1)) for a in (x, y, w))
            if self.noise_std > 0:
                x = x + self._rng.normal(0, self.noise_std, size=x.shape).astype(np.float32)
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)), dtype=np.float32)
        # flips/rot90 leave negative strides, which torch.from_numpy rejects
        y = np.ascontiguousarray((y > 0.5).astype(np.float32))
        w = np.ascontiguousarray(w.astype(np.float32))
        return x, y, w


# --------------------------------------------------------------------------------------
# high-level loaders used by train.py / inference.py
# --------------------------------------------------------------------------------------
def load_features_and_labels(feature_path=None, label_path=None, require_labels=True):
    """Back-compatible helper: returns (X(H,W,C), y(H,W), feat_meta, label_meta, tags)."""
    fp = resolve_path(feature_path, FEATURE_NAME_CANDIDATES)
    lp = None
    try:
        lp = resolve_path(label_path, LABEL_NAME_CANDIDATES)
    except FileNotFoundError:
        if require_labels:
            raise
    X, fmeta, tags = load_stack(fp)
    if lp is None:
        return X, None, fmeta, None, tags
    y, lmeta = load_labels(lp)
    if y.shape != X.shape[:2]:
        raise ValueError(f"label grid {y.shape} != feature grid {X.shape[:2]} - "
                         "the two rasters must share bounds/resolution (see problem page)")
    return X, y, fmeta, lmeta, tags
