#!/usr/bin/env python3
"""Blend parallel MC-fold predictions into one shaped submission.tif.

Inputs are fold directories produced by fold jobs (``python -m src.train`` with
``--override training.mc_id=F training.mc_splits=1`` followed by
``python -m src.inference --raw``).  Each fold dir must contain:

    prob_raw.tif      full-raster raw probability map from that fold's model
    heldout_mc*.npz   that fold's best-epoch held-out crop (pred, gt) — written by src/train.py
    manifest.json     that fold's run manifest (model name, per-arch dti, config)

Algorithm (this is the whole point — every step is derived from the official metric,
see docs/METRIC_STRATEGY.md):

  1. ensemble probability = nanmean of the fold maps (equal weights; the reference
     solution's MC ensemble is an equal-weight mean too);
  2. shaping calibration = for every (t0, thin) candidate, score each fold's held-out
     crop with THAT fold's model prediction and take the MEAN DTI across folds; choose
     the argmax.  One scalar pair fitted against |folds| x |windows| held-out pixels;
  3. submission = floor(t0) -> distance-R dominating thinning -> clip to [0,1];
     NaN outside the data footprint, written on the sample submission's grid;
  4. optional informational scoring vs the PUBLIC labels (known faults) — NOT the test
     set; reported as such.

Writes submission.tif + blend_report.json (auditable: calibration table, masses,
shas, grids).  Exit code 1 if the submission fails the format self-check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import compute_distance_weighted_tversky, score_arrays_blocked  # noqa: E402
from src.submission_optim import optimize_submission  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_folds(fold_dirs: list[str]):
    folds = []
    for d in fold_dirs:
        d = Path(d)
        prob = d / "prob_raw.tif"
        if not prob.exists():
            raise SystemExit(f"fold dir {d}: missing prob_raw.tif (run src/inference --raw there)")
        held = sorted(d.glob("heldout_mc*.npz"))
        manifest = d / "manifest.json"
        man = json.loads(manifest.read_text()) if manifest.exists() else {}
        with rasterio.open(prob) as src:
            arr = src.read(1).astype(np.float32)
            grid = dict(width=src.width, height=src.height, crs=str(src.crs),
                        transform=list(src.transform), res=[float(src.res[0]), float(src.res[1])])
        if held:
            z = np.load(held[0])
            pc, gc = z["pred"].astype(np.float32), z["gt"].astype(np.float32)
        else:
            pc = gc = None
        fdti = [m.get("dti") for m in man.get("models", [])]
        folds.append(dict(dir=str(d), prob=arr, pred_crop=pc, gt_crop=gc, grid=grid,
                          manifest=man, model=[m.get("file") for m in man.get("models", [])],
                          fold_dti=fdti,
                          mean_dti=(float(np.mean([f for f in fdti if f is not None]))
                                    if any(f is not None for f in fdti) else 0.0)))
    # grids must agree exactly — different bounds would silently misalign pixels
    for f in folds[1:]:
        if f["grid"] != folds[0]["grid"]:
            raise SystemExit(f"grid mismatch between fold dirs {folds[0]['dir']} and {f['dir']}")
    return folds


def calibrate_shaping(folds, R: int, thresholds: np.ndarray, alpha: float, beta: float, pre=None):
    """Pooled held-out search over (t0, thin). Returns (t0*, thin*, mean_dti*, table).

    Each fold contributes DTI(shaped fold-model crop, fold gt crop); the chosen point
    maximises the MEAN over folds.  The raw (unshaped) mean is reported too so the
    gain from shaping is auditable.
    """
    usable = [f for f in folds if f["pred_crop"] is not None]
    if not usable:
        return 0.3, True, float("nan"), []
    if pre is not None:
        # the FULL map gets `pre` before shaping, so calibration must see the same
        # transform on the fold crops - otherwise the floor is fitted to a different
        # distribution than the one it will be applied to (found by A/B, 2026-09-15:
        # post-Frangi calibration changed the whole outcome).
        for f in usable:
            f["pred_crop"] = pre(f["pred_crop"])
    table = []
    raw_mean = float(np.mean([compute_distance_weighted_tversky(f["pred_crop"], f["gt_crop"],
                                                                 R_pixels=R, alpha=alpha, beta=beta)
                              for f in usable]))
    best = (raw_mean, float(thresholds[0]), True, 0)
    for it, t in enumerate(thresholds):
        vals = []
        for thin in (False, True):
            v = float(np.mean([
                compute_distance_weighted_tversky(
                    optimize_submission(f["pred_crop"], R=R, t0=float(t), thin=thin, hard=True, gamma=1.0),
                    f["gt_crop"], R_pixels=R, alpha=alpha, beta=beta)
                for f in usable]))
            if v > best[0]:
                best = (v, float(t), thin, len(table))
            table.append(dict(t0=float(t), thin=bool(thin), mean_dti=v))
    table.insert(0, dict(t0=None, thin=None, mean_dti=raw_mean, raw=raw_mean))  # row 0 = unshaped
    return best[1], best[2], best[0], table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", required=True, help="fold directories (see module docstring)")
    ap.add_argument("--config", default="configs/config_ci_ensemble.yaml")
    ap.add_argument("--sample", default=None, help="sample_submission.tif — authoritative grid for the write")
    ap.add_argument("--labels", default=None, help="known-fault raster for the informational score")
    ap.add_argument("--out", default="submission.tif")
    ap.add_argument("--report", default=None, help="defaults to <out stem>_report.json")
    ap.add_argument("--frangi", action="store_true",
                    help="A/B knob: vesselness (Frangi) line enhancement on the blended mean map "
                         "BEFORE shaping (src/postprocess.frangi_enhance). Off by default until "
                         "measured to help on held-out calibration; see SUGGESTIONS.md.")
    ap.add_argument("--shaping-grid", type=int, default=None, help="threshold count (default: config)")
    ap.add_argument("--weights", choices=["equal", "dti"], default="equal",
                    help="fold averaging weights. 'dti' = softmax over each fold's best HELD-OUT "
                         "shaped DTI. MEASURED 2026-09-15 on the real 512x512 fixture window: with "
                         "2 folds a weak fold drowned the strong one and dti-weights helped "
                         "(0.0143 -> 0.0984); with 6 folds per-fold DTI differences are crop-noise "
                         "and dti-weights HURT (0.1033 -> 0.0656). Equal averaging is therefore the "
                         "default for the 6-fold workflow; consider 'dti' only to exclude a known-"
                         "catastrophic fold. Evidence: data/evidence/runs/local-mini-ensemble/.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    R = int(cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"])
    alpha = float(cfg["training"]["alpha"])
    beta = float(cfg["training"]["beta"])
    t0_cfg = time.time()

    folds = load_folds(args.folds)
    print(f"loaded {len(folds)} fold(s): {[f['dir'] for f in folds]}")

    # 1. ensemble mean (equal, or softmax over fold held-out DTI when --weights dti).
    # NaN-safe and footprint-aware: pixels where a fold is outside the data footprint
    # average over the folds that ARE present (weighted by their weights), exactly like
    # np.nanmean does for the equal case.
    stack = np.stack([f["prob"] for f in folds])
    ws = None
    if args.weights == "dti" and all(f["mean_dti"] > 0 for f in folds):
        ws = np.array([f["mean_dti"] for f in folds], dtype=np.float64)
        ws = np.exp((ws - ws.max()) * 8.0)
        ws = ws / ws.sum()
        print(f"fold weights (softmax over held-out DTI): {[round(w, 3) for w in ws]}")
    valid = ~np.isnan(stack)
    allnan = ~valid.any(axis=0)
    vals = np.where(valid, stack, 0.0)
    if ws is None:
        num = vals.sum(axis=0)
        present = valid.sum(axis=0).astype(np.float64)
    else:
        num = np.tensordot(ws, vals, axes=1)              # sum_i w_i p_i over present folds
        present = np.tensordot(ws, valid.astype(np.float64), axes=1)
    mean = num / np.maximum(present, 1e-9)
    mean[allnan] = np.nan
    mean = np.clip(np.nan_to_num(mean, nan=0.0), 0.0, 1.0)   # 0 outside footprint for now
    pre_mass = float(mean[~allnan].sum())

    enh = None
    if args.frangi:
        from src.postprocess import frangi_enhance
        enh = lambda m: frangi_enhance(m, scale_range=(1, 6), scale_step=2, weight=0.35)
        mean = enh(mean)
        print("frangi line-enhancement applied to the blended map (A/B, consistent calibration)")

    # 2. pooled held-out shaping calibration ---------------------------------------
    n_grid = int(args.shaping_grid or cfg["training"].get("shaping_grid", 11))
    thr = np.linspace(0.02, 0.9, n_grid)
    t0b, thinb, mean_dti, table = calibrate_shaping(folds, R, thr, alpha=alpha, beta=beta, pre=enh)
    print(f"pooled shaping: t0={t0b:.3f} thin={thinb} -> mean held-out DTI {mean_dti:.4f}"
          f" (unshaped {table[0]['mean_dti']:.4f})")

    # 3. shape on the full ensemble map --------------------------------------------
    q = optimize_submission(mean, R=R, t0=t0b, thin=bool(thinb), hard=True, gamma=1.0)
    q = np.clip(q, 0.0, 1.0).astype(np.float32)
    post_mass = float(q[~allnan].sum())
    if post_mass <= 0.0:
        # Observed failure mode (fixture A/B, 2026-09-15): a pre-transform that flattens
        # the map can drive EVERY pixel under the calibrated floor -> an all-zero
        # submission (DTI 0).  Never write that silently; fail the step loudly instead.
        raise SystemExit("BLEND ABORTED: shaping collapsed the map to all zeros "
                         f"(t0={t0b:.3f}, thin={bool(thinb)}, mean mass {pre_mass:.0f}). "
                         "A pre-transform likely changed the distribution outside calibration.")
    q[allnan] = np.nan                                        # spec: outside bounds null/nan

    # write on the sample grid when given, else on the fold grid (identical by spec;
    # scripts/validate_submission.py re-checks against both the sample and the features)
    grid = folds[0]["grid"]
    if args.sample and Path(args.sample).exists():
        with rasterio.open(args.sample) as src:
            sgrid = dict(width=src.width, height=src.height, crs=str(src.crs),
                         transform=list(src.transform), res=[float(src.res[0]), float(src.res[1])])
            profile = src.profile.copy()
        if sgrid["width"] != grid["width"] or sgrid["height"] != grid["height"]:
            print(f"WARNING sample grid {sgrid['width']}x{sgrid['height']} != fold grid "
                  f"{grid['width']}x{grid['height']} — writing on the SAMPLE grid (crop/pad)")
            buf = np.full((sgrid["height"], sgrid["width"]), np.nan, np.float32)
            hh, ww = min(sgrid["height"], q.shape[0]), min(sgrid["width"], q.shape[1])
            buf[:hh, :ww] = q[:hh, :ww]
            q = buf
        profile.update(driver="GTiff", count=1, dtype="float32", nodata=None,
                       compress="lzw", TILED="YES")
    else:
        profile = dict(driver="GTiff", height=grid["height"], width=grid["width"], count=1,
                       dtype="float32", crs=grid["crs"],
                       transform=rasterio.Affine(*grid["transform"]),
                       nodata=None, compress="lzw", TILED="YES")

    out = Path(args.out)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(q, 1)
        dst.set_band_description(1, "fault-presence probability (GEMSDOE 6-fold MC ensemble, shaped)")
        dst.update_tags(source="GEMSDOE train-ensemble workflow",
                        n_models=str(len(folds)),
                        models=";".join(str(m) for f in folds for m in f["model"]),
                        shaping_t0=str(t0b), shaping_thin=str(bool(thinb)))
    print(f"wrote {out}  nonzero={int(np.count_nonzero(np.nan_to_num(q)))} "
          f"finite={int(np.isfinite(q).sum())}/{q.size} mass {pre_mass:.0f} -> {post_mass:.0f}")

    # 4. informational score vs known (public) faults -------------------------------
    local = None
    if args.labels and Path(args.labels).exists():
        with rasterio.open(args.labels) as src:
            lab = src.read(1)
            if src.nodata is not None:
                lab[lab == src.nodata] = 0
        gt = (np.asarray(lab, np.float32) > 0.5).astype(np.float32)
        if gt.shape != q.shape:
            gt = gt[: q.shape[0], : q.shape[1]]
        d, (tp, fp, fn) = score_arrays_blocked(np.nan_to_num(q), gt, R_pixels=R, alpha=alpha, beta=beta)
        n_g = float(gt.sum())
        closed = tp / ((1 - beta) * tp + alpha * fp + beta * n_g + cfg["metric"].get("epsilon", 1e-7))
        ones = np.ones_like(gt)
        d_ones, (tp1, fp1, fn1) = score_arrays_blocked(ones, gt, R_pixels=R, alpha=alpha, beta=beta)
        local = dict(dti_known_faults=float(d), TP_w=float(tp), FP_w=float(fp), FN_w=float(fn),
                     closed_form=float(closed), closed_form_matches=bool(abs(closed - d) < 1e-6),
                     blanket_ones_dti=float(d_ones),
                     note="scored against the PUBLIC known-fault raster — NOT the competition test "
                          "set (private expert-labelled NEW faults). Optimistic AND wrong-universe: "
                          "reported for pipeline monitoring only.")
        print(f"informational DTI vs known faults = {d:.4f} (blanket-ones floor {d_ones:.4f})")

    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/blend_submission.py",
        minutes=round((time.time() - t0_cfg) / 60.0, 2),
        n_folds=len(folds),
        folds=[dict(dir=f["dir"], models=f["model"], fold_best_dti=f["fold_dti"],
                    grid=f["grid"]) for f in folds],
        metric=dict(R_pixels=R, alpha=alpha, beta=beta),
        shaping=dict(t0=t0b, thin=bool(thinb), mean_heldout_dti=float(mean_dti),
                     unshaped_mean_heldout_dti=float(table[0]["mean_dti"]),
                     calibration_table=table),
        probability_mass=dict(pre_shaping=pre_mass, post_shaping=post_mass,
                              collapse_factor=(pre_mass / post_mass if post_mass > 0 else None)),
        fold_weights=("equal" if ws is None else [round(float(w), 4) for w in ws]),
        submission=dict(path=str(out), bytes=out.stat().st_size, sha256=sha256(out),
                        grid=grid, finite_px=int(np.isfinite(q).sum()), total_px=int(q.size),
                        nonzero_px=int(np.count_nonzero(np.nan_to_num(q))),
                        min=float(np.nanmin(q)), max=float(np.nanmax(q)),
                        mean=float(np.nanmean(q))),
        local_score_vs_known=local,
    )
    rp = Path(args.report) if args.report else out.with_name(out.stem + "_report.json")
    rp.write_text(json.dumps(report, indent=1))
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
