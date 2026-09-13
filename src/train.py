"""Training for the GEMS Prize - Monte-Carlo CV ensemble trained on the *actual* metric.

Design decisions, each traceable to a verified source:
* metric = distance-weighted Tversky (alpha=0.2, beta=0.8, R=300 m) - problem page
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
  -> we optimise a differentiable version of that exact metric (src/losses.py), plus BCE for
     dense gradients, instead of the reference solution's plain TverskyLoss.
* Monte-Carlo cross-validation (5 splits) and 128-px patches are the reference solution's
  design (cell 16: MC=5, patch_size=128, test_proportion=0.5, batch 32, epochs 5,
  AdamW lr=1e-4, resnet18 U-Net).  We keep the *structure* (so results are comparable) and
  raise capacity/schedule for the leaderboard run in configs/config.yaml.
* model selection uses the metric computed on the *stitched global* held-out test patches
  (not a mean of per-patch scores), because DTI is not decomposable over patches: the
  official scorer evaluates one continuous raster.
* every artefact needed to reproduce is written to outputs/manifest.json
  (Official Rules 3.5: assets must "sufficiently reproduce the winning results").
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (FaultDataset, band_names, load_features_and_labels, load_norm_stats,
                      make_patches, apply_norm_stats, fit_norm_stats, save_norm_stats)
from .losses import CombinedLoss, TverskyLoss, DistanceWeightedTverskyLoss
from .metrics import compute_distance_weighted_tversky, score_arrays_blocked
from .models import count_params, get_model
from .submission_optim import search_threshold


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_patches(model, X, device, batch_size=16):
    model.eval()
    out = []
    ds = torch.utils.data.TensorDataset(torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).float())
    for (xb,) in DataLoader(ds, batch_size=batch_size):
        logits = model(xb.to(device))
        out.append(torch.sigmoid(logits)[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


def stitch(patches, origins, shape, crop=None):
    """Place non-overlapping patches back into a global map.

    `shape` is the *padded* grid (windows are aligned to it); `crop` trims back to the
    original raster extent, mirroring the reference solution's
    `y_final = y_combined[:y_orig.shape[0], :y_orig.shape[1]]`.
    """
    out = np.zeros(shape, np.float32)
    p = patches.shape[1]
    for arr, (i, j) in zip(patches, origins):
        out[i:i + p, j:j + p] = arr
    if crop is not None:
        out = out[: crop[0], : crop[1]]
    return out


def heldout_maps(model, res, y_shape, device, cfg, R):
    """Held-out (Monte-Carlo test) predictions/GT: full stitched map + search crop.

    Returns (pred_full, gt_full, pred_crop, gt_crop).  `pred_full`/`gt_full` are what the
    reported DTI uses; the cropped pair bounds the cost of the shaping grid search.
    """
    probs = predict_patches(model, res["X_test"], device, batch_size=cfg["inference"]["batch_size"])
    Hp, Wp = res["summary"]["H"], res["summary"]["W"]
    tw = res["summary"]["test_windows"]
    pred_full = stitch(probs, tw, (Hp, Wp), crop=y_shape)
    gt_full = stitch(res["y_test"], tw, (Hp, Wp), crop=y_shape)
    nmax = int(cfg["training"].get("shaping_windows", 48))
    box = tw[:nmax]
    if not box:
        return pred_full, gt_full, pred_full, gt_full
    p_ = int(cfg["training"]["patch_size"])
    y0, y1 = min(i for i, j in box), max(i for i, j in box) + p_
    x0, x1 = min(j for i, j in box), max(j for i, j in box) + p_
    return pred_full, gt_full, pred_full[y0:y1, x0:x1], gt_full[y0:y1, x0:x1]


def shaped_table(pg, gg, R, thresholds):
    """DTI of the *shaped* submission for every (t0, thin) candidate (src/submission_optim)."""
    from .submission_optim import optimize_submission
    rows = []
    for t in thresholds:
        for thin in (False, True):
            q = optimize_submission(pg, R=R, t0=float(t), thin=bool(thin))
            rows.append((float(t), bool(thin), float(compute_distance_weighted_tversky(q, gg, R_pixels=R))))
    return rows



def _score_full(pred_full, gt_full, cfg, R):
    dti, (tp, fp, fn) = score_arrays_blocked(np.nan_to_num(pred_full), gt_full, R_pixels=R,
                                             alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"])
    return dti, None, dict(TP_w=tp, FP_w=fp, FN_w=fn)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, fpw_enabled, sched=None):
    model.train()
    tot, nb = 0.0, 0
    for X, y, w in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(X)
                loss = criterion(logits, y, fp_weight=w.to(device) if fpw_enabled else None)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y, fp_weight=w.to(device) if fpw_enabled else None)
            loss.backward()
            optimizer.step()
        if sched is not None:
            sched.step()          # OneCycleLR steps per batch, not per epoch
        tot += float(loss)
        nb += 1
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for o in args.override:
        k, v = o.split("=", 1)
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        try:
            cur[parts[-1]] = json.loads(v)
        except json.JSONDecodeError:
            cur[parts[-1]] = v

    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["training"].get("seed", 42)))
    device = pick_device()
    print(f"device={device}  config={args.config}")

    # ---------------------------------------------------------------- data ---------
    X, y, fmeta, lmeta, tags = load_features_and_labels(cfg["data"].get("feature_path"),
                                                        cfg["data"].get("label_path"))
    names = band_names(tags, X.shape[-1])
    print(f"features {X.shape} labels {y.shape}  bands={names[:4]}{'...' if len(names) > 4 else ''}")

    stats_path = out_dir / "norm_stats.json"
    if stats_path.exists():
        nstats = load_norm_stats(stats_path)
        print(f"reusing normalisation stats from {stats_path}")
    else:
        nstats = fit_norm_stats(X, tuple(cfg["data"].get("clip_percentile", (1.0, 99.0))))
        save_norm_stats(stats_path, nstats)
    Xn = apply_norm_stats(X, nstats, mode=cfg["data"].get("norm_mode", "clip_zscore"))

    R_px = int(cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"])
    use_fpw = bool(cfg["training"].get("use_global_fp_weight", True))

    manifest = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=cfg, band_names=names, n_bands=int(X.shape[-1]),
        feature_grid=list(X.shape[:2]), label_grid=list(y.shape),
        norm_stats_file=str(stats_path), R_pixels=R_px,
        cuda=torch.cuda.is_available(), torch=torch.__version__,
        feature_file=str(cfg["data"].get("feature_path")), models=[],
    )

    loss_name = cfg["training"].get("loss", "combined_dw")
    criterion = (CombinedLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"], R=R_px)
                 if loss_name == "combined_dw" else
                 DistanceWeightedTverskyLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"], R=R_px)
                 if loss_name == "dw_tversky" else
                 TverskyLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"]))

    hist = []
    calib_maps = []
    for mc in range(int(cfg["training"]["mc_splits"])):
        print(f"\n=== MC split {mc + 1}/{cfg['training']['mc_splits']} ===")
        res = make_patches(
            Xn, y, patch_size=cfg["training"]["patch_size"], train_step=cfg["training"]["train_step"],
            test_proportion=cfg["training"]["test_proportion"], seed=mc * 10 + int(cfg["training"].get("seed", 42)),
            neg_fraction=cfg["training"].get("neg_fraction", 0.35), R_pixels=R_px,
        )
        print("patches:", {k: v for k, v in res["summary"].items() if not k.endswith("_windows")})

        good = cfg["training"].get("good_channels")
        if good:
            res["X_train"] = res["X_train"][..., good]
            res["X_test"] = res["X_test"][..., good]
        in_ch = res["X_train"].shape[-1]

        archs = cfg["model"]["architectures"]
        arch = archs[mc % len(archs)]
        enc = cfg["model"].get("segformer_encoder", "mit_b2") if "segformer" in arch else cfg["model"]["encoder"]
        model = get_model(arch=arch, encoder=enc, in_channels=in_ch, classes=1,
                          pretrained=bool(cfg["model"].get("pretrained", False))).to(device)
        print(f"arch={arch} encoder={enc} in_ch={in_ch} params={count_params(model):,}")

        bs = cfg["training"]["batch_size"]
        train_ds = FaultDataset(res["X_train"], res["y_train"], res["fpw_train"] if use_fpw else None,
                                train=True, augment=bool(cfg["training"].get("augment", True)),
                                noise_std=float(cfg["training"].get("noise_std", 0.01)),
                                rand_crop_scale=tuple(cfg["training"].get("rand_crop_scale", (1.0, 1.0))),
                                seed=mc)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=int(cfg["training"].get("num_workers", 0)),
                              pin_memory=(device.type == "cuda"), drop_last=False)
        # test loader: no augmentation, no fpw needed (scored globally)
        test_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(np.ascontiguousarray(res["X_test"])).permute(0, 3, 1, 2).float(),
            torch.from_numpy(res["y_test"]).float())

        opt = optim.AdamW(model.parameters(), lr=float(cfg["training"]["init_lr"]),
                          weight_decay=float(cfg["training"]["weight_decay"]))
        epochs = int(cfg["training"]["epochs"])
        steps_per_epoch = max(1, len(train_dl))
        sched = None
        if cfg["training"].get("scheduler", "onecycle") == "onecycle":
            sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["training"]["init_lr"]) * 10,
                                                   total_steps=epochs * steps_per_epoch, pct_start=0.15)
        use_amp = device.type == "cuda" and bool(cfg["training"].get("amp", True))
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        best = dict(dti=-1.0, epoch=-1)
        best_state = None
        best_maps = None
        for ep in range(epochs):
            t0 = time.time()
            tr = train_one_epoch(model, train_dl, criterion, opt, device, scaler, use_amp,
                                 use_fpw and loss_name != "plain_tversky", sched=sched)
            pf, gf, pc, gc = heldout_maps(model, res, y.shape, device, cfg, R_px)
            dti_g, _unused, comps = _score_full(pf, gf, cfg, R_px)
            _thr = np.linspace(0.02, 0.9, int(cfg["training"].get("shaping_grid", 15)))
            tbl = shaped_table(pc, gc, R_px, _thr)
            dti_shaped, sh_best = (max(r[2] for r in tbl), max(tbl, key=lambda r: r[2]))
            print(f"epoch {ep + 1}/{epochs}  loss={tr:.4f}  DTI_raw={dti_g:.4f}  "
                  f"DTI_shaped={dti_shaped:.4f} (t0={sh_best[0]:.2f},thin={sh_best[1]})  "
                  f"TP={comps['TP_w']:.0f} FP={comps['FP_w']:.0f} "
                  f"FN={comps['FN_w']:.0f}  [{time.time() - t0:.0f}s]")
            hist.append(dict(mc=mc, epoch=ep + 1, loss=tr, dti_raw=dti_g, dti_shaped=dti_shaped,
                             **comps))
            # model selection on the SHAPED metric: that is the object that gets scored
            if dti_shaped > best["dti"]:
                best = dict(dti=dti_shaped, epoch=ep + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_maps = (pc.copy(), gc.copy())
        if cfg["training"].get("calibrate_shaping", True) and best_maps is not None:
            calib_maps.append(best_maps)            # pooled search over all splits, below

        ck = out_dir / f"model_mc{mc}_{arch}_{enc}.pt"
        torch.save(best_state, ck)
        manifest["models"].append(dict(file=ck.name, arch=arch, encoder=enc, in_channels=int(in_ch),
                                       classes=1, dti=best["dti"], epoch=best["epoch"], mc=mc,
                                       patch_size=cfg["training"]["patch_size"],
                                       test_windows=res["summary"]["test_windows"]))
        print(f"saved {ck} (best DTI {best['dti']:.4f} @ epoch {best['epoch']})")

    # ---- pooled shaping calibration ------------------------------------------------
    # choose ONE (t0, thin) maximising the MEAN held-out DTI across all MC splits: a single
    # scalar fitted against |splits| x |windows| pixels of held-out truth, so this is far
    # from the overfitting risk of per-split tuning, and it is what inference.py applies.
    if calib_maps and cfg["training"].get("calibrate_shaping", True):
        thr = np.linspace(0.02, 0.9, int(cfg["training"].get("shaping_grid", 15)))
        tables = [shaped_table(pg, gg, R_px, thr) for pg, gg in calib_maps]
        mean_dti = np.mean([np.array([r[2] for r in t]) for t in tables], axis=0)
        i = int(np.argmax(mean_dti))
        raw = float(np.mean([compute_distance_weighted_tversky(pg, gg, R_pixels=R_px) for pg, gg in calib_maps]))
        t0b, thinb, _ = tables[0][i]
        manifest["shaping"] = dict(t0=float(t0b), thin=bool(thinb), hard=True, gamma=1.0,
                                   mean_heldout_dti=float(mean_dti[i]), mean_heldout_dti_raw=raw,
                                   n_splits_used=len(calib_maps),
                                   search=[dict(t0=float(r[0]), thin=bool(r[1]), mean_dti=float(mean_dti[k]))
                                           for k, r in enumerate(tables[0])],
                                   note="floor+thin by pooled held-out mean DTI; src/submission_optim.py")
        print(f"\npooled shaping: t0={t0b:.3f} thin={thinb} -> held-out DTI {mean_dti[i]:.4f} "
              f"(raw probabilities: {raw:.4f})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_dir / "train_history.json").write_text(json.dumps(hist, indent=1))
    print(f"\nTraining done. manifest -> {out_dir / 'manifest.json'}")
    if hist:
        print(f"mean best DTI across splits: "
              f"{np.mean([m['dti'] for m in manifest['models']]):.4f}  (local, test-window stitched)")


if __name__ == "__main__":
    main()
