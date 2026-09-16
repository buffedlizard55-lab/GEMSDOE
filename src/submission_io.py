"""Torch-free GeoTIFF submission writer with a fail-loud post-write verification.

WHY THIS MODULE EXISTS (defect found 2026-09-16, evidence in this repo)
----------------------------------------------------------------------
The 6-fold ensemble workflow (Actions run 35042805806) trained all six folds, then died at
the LAST step with

    rasterio._err.CPLE_AppDefinedError: _TIFFVSetField:submission.tif: Bad value 3292 for "TileWidth" tag
    rasterio.errors.RasterBlockError: The height and width of TIFF dataset blocks must be multiples of 16

Root cause: the writer copied the *sample submission's* rasterio profile and then forced
``TILED="YES"``.  A striped GeoTIFF reports ``blockxsize == width`` (3292 here); GDAL then
refuses to reinterpret 3292 as a tile width, because TIFF tile dimensions must be multiples
of 16.  Nothing was written, the workflow still reported success (the crash was masked by a
``| tee`` pipeline without ``pipefail``), and a 110-byte GDAL stub was committed to the
branch as if it were a submission.

So a submission writer has exactly two jobs, and this module does both:
  1. build a *clean* profile - never inherit block geometry from another file;
  2. prove the file it just wrote is readable, correctly gridded and non-degenerate, and
     raise if it is not (an empty/invalid submission must never look like success).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

__all__ = ["TILE", "clean_profile", "write_submission", "sha256_file"]

TILE = 256  # multiple of 16 -> always a legal TIFF tile dimension

# keys rasterio copies from a source dataset that must NOT be reused verbatim when the
# block layout changes (they describe the *source* file's storage, not our output's)
_STALE_KEYS = ("blockxsize", "blockysize", "blockyshapes", "strides")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_profile(src_profile: dict | None = None, *, height: int | None = None,
                  width: int | None = None, crs=None, transform=None,
                  dtype: str = "float32", compress: str = "lzw",
                  tiled: bool = True, tile: int = TILE, nodata=None) -> dict:
    """Build a legal GTiff profile for a submission raster.

    ``src_profile`` (e.g. from ``sample_submission.tif``) contributes georeferencing and
    layout *intent* only - never its block geometry.  Tile blocks are pinned to a multiple
    of 16 for every axis, which is what the failed run got wrong.
    """
    prof: dict = {}
    if src_profile:
        prof.update({k: v for k, v in src_profile.items() if k not in _STALE_KEYS})
        prof["driver"] = "GTiff"
    prof.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress=compress)
    if height is not None:
        prof["height"] = int(height)
    if width is not None:
        prof["width"] = int(width)
    if crs is not None:
        prof["crs"] = crs
    if transform is not None:
        prof["transform"] = transform
    for k in _STALE_KEYS:
        prof.pop(k, None)
    if tiled:
        prof["tiled"] = True
        # a raster narrower/shorter than the tile is legal (GDAL pads), but the tile
        # dimension itself must stay a multiple of 16
        prof["blockxsize"] = int(tile)
        prof["blockysize"] = int(tile)
    else:
        prof["tiled"] = False
    prof.pop("interleave", None)
    return prof


def write_submission(path, array: np.ndarray, profile: dict, *, blockcheck: bool = True,
                     band_description: str | None = None, tags: dict | None = None) -> dict:
    """Write ``array`` to ``path`` and verify the bytes on disk.

    Returns a verification dict (path, bytes, sha256, grid, dtype, counts, ranges).
    Raises ``RuntimeError`` - never returns quietly - if the written file does not read
    back as a single-band float32 raster on the requested grid with any finite mass.
    """
    path = Path(path)
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"submission array must be 2-D, got shape {arr.shape}")
    h, w = arr.shape
    if int(profile.get("height", h)) != h or int(profile.get("width", w)) != w:
        raise ValueError(f"profile {profile.get('width')}x{profile.get('height')} "
                         f"does not match array {w}x{h}")
    arr = arr.astype(np.float32, copy=False)
    bad = np.isfinite(arr) & ((arr < -1e-6) | (arr > 1 + 1e-6))
    if bad.any():
        raise ValueError(f"values outside [0,1] (n={int(bad.sum())}, "
                         f"max={float(np.nanmax(arr))}) - spec requires 0..1")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        # rasterio has no append mode ("a" raises ValueError), so tags must be set here
        if band_description:
            dst.set_band_description(1, band_description)
        if tags:
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"writer produced no bytes at {path}")

    # ---- read the bytes back: the step that would have caught the 110-byte stub --------
    with rasterio.open(path) as src:
        if src.width != w or src.height != h:
            raise RuntimeError(f"read-back grid {src.width}x{src.height} != written {w}x{h}")
        if src.count != 1:
            raise RuntimeError(f"read-back has {src.count} bands, expected 1")
        if src.dtypes[0] != "float32":
            raise RuntimeError(f"read-back dtype {src.dtypes[0]}, expected float32")
        back = src.read(1)
        if (src.crs is None) and (profile.get("crs") is not None):
            raise RuntimeError("read-back lost the CRS")
        if blockcheck and src.block_shapes and src.profile.get("tiled"):
            for bh, bw in src.block_shapes:
                if bh % 16 or bw % 16:
                    raise RuntimeError(f"illegal tiled block {bh}x{bw} (must be multiples of 16)")
        finite = int(np.isfinite(back).sum())
        info = dict(
            path=str(path), bytes=int(path.stat().st_size), sha256=sha256_file(path),
            width=int(src.width), height=int(src.height), count=int(src.count),
            dtype=str(src.dtypes[0]), crs=str(src.crs), res=[float(r) for r in src.res],
            transform=list(src.transform), nodata=src.nodata,
            tiled=bool(src.profile.get("tiled")),
            block_shapes=[list(b) for b in (src.block_shapes or [])],
            finite_px=finite, total_px=int(back.size), nan_px=int(back.size - finite),
            nonzero_px=int(np.count_nonzero(np.nan_to_num(back))),
            prob_mass=float(np.nansum(back)),
        )
    if info["finite_px"] == 0:
        raise RuntimeError("read-back raster has no finite pixels")
    if info["nonzero_px"] == 0:
        raise RuntimeError("read-back raster is all zeros (total-fault-absence template) - "
                           "refusing to report success")
    return info


def dump_json(obj, path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    return str(p)
