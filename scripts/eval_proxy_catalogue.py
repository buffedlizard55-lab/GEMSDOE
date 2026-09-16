#!/usr/bin/env python3
"""Score a prediction against the proxy catalogue - the closest available stand-in for the scored
"new fault" population.

WHY THIS EXISTS
---------------
Every number this repository has published about submission *quality* was measured against
`labels.tif`, i.e. the faults the catalogue already contains.  The rules say both prize phases score
faults it does NOT contain (§1.1, §3.3; verbatim quotes in `scripts/verify_rules_quotes.py`).  A
metric on the wrong universe cannot choose between a model that generalises and one that memorises.

`scripts/fetch_proxy_faults.py` + `scripts/build_proxy_catalogue.py` build an independent fault set
(USGS SGMC, Horton et al. 2017, DOI 10.3133/ds1052) rasterised on the exact competition grid, split
into "the labels already contain it" (code 1) and "the labels do not" (code 2).  This script scores
against the code-2 pixels with the OFFICIAL metric - same R, alpha, beta, kernel, same call into
`src.metrics.GtContext` that the training loop and the submission calibration use.

WHAT THE NUMBER MEANS, AND WHAT IT DOES NOT
-------------------------------------------
* It is a *population* measurement on real mapped faults absent from the training labels.  A model
  that reproduces the catalogue scores exactly 0 on it, because code-2 pixels are >R from every
  labelled pixel (that identity is asserted here, not assumed).
* It is NOT the competition metric: the scored faults were drawn by experts from the GeoDAWN
  geophysics; SGMC faults were drawn by state-map geologists from surface mapping.  Neither
  population contains the other.  Use it to compare *policies* (models, floors, emission widths),
  never to predict a leaderboard position.
* The FP term is a global sum, so a prediction with mass far from the proxy faults is penalised
  exactly as the official metric would penalise it.  Nothing is cropped by default.

USAGE
    python scripts/eval_proxy_catalogue.py --pred submission.tif \
        --proxy data/evidence/proxy/proxy_catalogue.tif --labels data/labels.tif \
        --out data/evidence/proxy/eval_submission.json
    # policy sweep (floor x thinning x emission width) on an ensemble probability raster:
    python scripts/eval_proxy_catalogue.py --pred ensemble_mean.tif --sweep --shaping-grid 5 \
        --dilate-grid 0,1,2,3,4,6 --out data/evidence/proxy/eval_sweep.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import GtContext                                             # noqa: E402
from src.submission_optim import optimize_submission, shaping_thresholds      # noqa: E402

CODE_NEAR, CODE_ONLY = 1, 2
DEFAULTS = dict(R=3, alpha=0.2, beta=0.8, eps=1e-7)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_grid(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        meta = {"width": src.width, "height": src.height, "crs": str(src.crs),
                "transform": [round(v, 6) for v in list(src.transform)[:6]],
                "dtype": src.dtypes[0], "nodata": src.nodata}
    return arr, meta


def same_grid(a: dict, b: dict) -> bool:
    return (a["width"], a["height"]) == (b["width"], b["height"]) and a["crs"] == b["crs"]


def score(pred: np.ndarray, ctx: GtContext, R: int, alpha: float, beta: float, eps: float) -> dict:
    dti, (tp, fp, fn) = ctx.score(pred, alpha=alpha, beta=beta, eps=eps, return_components=True)
    return {"dti": round(float(dti), 6), "TP_w": round(float(tp), 3), "FP_w": round(float(fp), 3),
            "FN_w": round(float(fn), 3), "mass": round(float(np.nansum(pred)), 1),
            "emission_px": int(np.count_nonzero(np.nan_to_num(pred)))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="prediction raster (submission.tif, or a probability map)")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--labels", default=None, help="known-fault raster (needed for --truth combined)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", choices=["only", "all", "combined"], default="only",
                    help="only = proxy faults absent from the catalogue (default); all = every proxy "
                         "fault; combined = proxy + labelled faults")
    ap.add_argument("--sweep", action="store_true",
                    help="also search floor x thinning x emission width on THIS population")
    ap.add_argument("--shaping-grid", type=int, default=5)
    ap.add_argument("--dilate-grid", default="0,1,2,3,4,6")
    ap.add_argument("--thin-off", action="store_true", help="include thin=False rows in the sweep")
    ap.add_argument("--R", type=int, default=DEFAULTS["R"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    ap.add_argument("--eps", type=float, default=DEFAULTS["eps"])
    a = ap.parse_args()

    pred, pmeta = read_grid(Path(a.pred))
    coded, cmeta = read_grid(Path(a.proxy))
    if coded.dtype != np.uint8 and coded.max() <= 3:
        coded = coded.astype("uint8")
    if not same_grid(pmeta, cmeta):
        raise SystemExit(f"grid mismatch: {a.pred} {pmeta} vs {a.proxy} {cmeta}")

    labels = None
    if a.labels:
        labels, lmeta = read_grid(Path(a.labels))
        if not same_grid(lmeta, cmeta):
            raise SystemExit(f"grid mismatch: {a.labels} vs {a.proxy}")

    near, only = coded == CODE_NEAR, coded == CODE_ONLY
    if a.truth == "only":
        truth = only
    elif a.truth == "all":
        truth = (coded > 0)
    else:
        if labels is None:
            raise SystemExit("--truth combined needs --labels")
        truth = (coded > 0) | (labels > 0.5)
    n_truth = int(truth.sum())
    if n_truth == 0:
        raise SystemExit("empty truth set - refusing to produce a number from it")

    ctx = GtContext(truth.astype("float32"), R_pixels=a.R)
    res: dict = {"as_submitted": score(pred, ctx, a.R, a.alpha, a.beta, a.eps)}

    # ---- baselines: each one answers "could this number be earned trivially?" -------------------
    footprint = np.isfinite(pred) if np.isfinite(pred).any() else np.ones_like(pred, dtype=bool)
    baselines = {"zeros": np.zeros_like(pred),
                 "blanket_ones": footprint.astype("float32")}
    if labels is not None:
        lab = np.where(np.isfinite(labels), labels, 0.0).astype("float32")
        baselines["catalogue_copy"] = (lab > 0.5).astype("float32")
        baselines["catalogue_plus_submission"] = np.maximum((lab > 0.5).astype("float32"),
                                                            (np.nan_to_num(pred) > 0).astype("float32"))
    baselines["submission_skeleton_dilated_6px"] = None      # filled by the sweep when requested
    res["baselines"] = {k: score(v, ctx, a.R, a.alpha, a.beta, a.eps)
                        for k, v in baselines.items() if v is not None}

    # Acceptance property, asserted rather than assumed: on the proxy-only truth a perfect
    # reproduction of the catalogue must score EXACTLY zero (every truth pixel is >R from a label).
    if "catalogue_copy" in res["baselines"]:
        cc = res["baselines"]["catalogue_copy"]["dti"]
        res["acceptance"] = {
            "catalogue_copy_dti_on_proxy_only": cc,
            "expected": 0.0,
            "passed": bool(abs(cc) < 1e-9),
            "meaning": ("a catalogue-copy submission earns nothing here, so this population is not "
                        "a restatement of the training labels"),
        }
        if a.truth == "only" and not res["acceptance"]["passed"]:
            print("FAIL: catalogue copy scored non-zero on the proxy-only truth - the proxy or the "
                  "grid alignment is broken; the number must not be used", file=sys.stderr)

    # ---- policy sweep: does any shaping beat the current one ON THIS POPULATION? ----------------
    sweep_table: list[dict] = []
    if a.sweep:
        dilates = tuple(sorted({int(v) for v in str(a.dilate_grid).split(",") if v.strip()}))
        for t0 in shaping_thresholds(a.shaping_grid):
            for thin in ((False, True) if a.thin_off else (True,)):
                for d in (dilates if thin else (0,)):
                    q = optimize_submission(pred, R=a.R, t0=float(t0), thin=bool(thin),
                                            hard=True, gamma=1.0, dilate=int(d))
                    row = {"t0": float(t0), "thin": bool(thin), "dilate": int(d)}
                    row.update(score(q, ctx, a.R, a.alpha, a.beta, a.eps))
                    row["mean_kept_px"] = int(np.count_nonzero(q))
                    sweep_table.append(row)
                    print(f"  t0={t0:<8.5g} thin={int(thin)} dilate={d}px -> proxy DTI "
                          f"{row['dti']:.4f}  kept {row['mean_kept_px']} px")
        sweep_table.sort(key=lambda r: -r["dti"])
        res["shaping_sweep"] = sweep_table
        res["sweep_best"] = sweep_table[0] if sweep_table else None
        by_width = {}
        for r in sweep_table:
            if r["thin"]:
                by_width.setdefault(r["dilate"], []).append(r["dti"])
        res["best_per_emission_width"] = {str(k): round(max(v), 6) for k, v in sorted(by_width.items())}
        res["sweep_verdict"] = {
            "skeleton_dti": res["best_per_emission_width"].get("0"),
            "best_dti": res["sweep_best"]["dti"] if res["sweep_best"] else None,
            "acceptance_rule": ("change the default shaping only if a candidate beats the current "
                                "default by > 0.01 proxy DTI here AND is reproduced on a second "
                                "ensemble; this population is the closest measurable stand-in for "
                                "the scored one, not the scored one"),
        }

    sweep_px = int(np.count_nonzero(np.nan_to_num(pred)))
    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/eval_proxy_catalogue.py",
        "purpose": ("score a prediction against faults the training labels do not contain "
                    "(docs/DISCOVERY_PLAN.md 3b) - the only local measurement aligned with the "
                    "scored 'new fault' population"),
        "inputs": {
            "pred": a.pred, "pred_sha256": sha256(Path(a.pred)), "pred_meta": pmeta,
            "proxy": a.proxy, "proxy_sha256": sha256(Path(a.proxy)), "proxy_meta": cmeta,
            "labels": a.labels, "truth_mode": a.truth,
            "metric": {"R_pixels": a.R, "R_meters": a.R * 100, "alpha": a.alpha, "beta": a.beta,
                       "eps": a.eps},
        },
        "truth": {"mode": a.truth, "px": n_truth, "km": round(n_truth * 0.01, 3),
                  "proxy_only_px": int(only.sum()), "proxy_near_label_px": int(near.sum()),
                  "catalogue_coverage_of_proxy": round(
                      int(near.sum()) / max(int(near.sum()) + int(only.sum()), 1), 4)},
        "prediction": {"emission_px": sweep_px, "mass": round(float(np.nansum(pred)), 1)},
        "results": res,
        "caveats": [
            "Population, not score: SGMC faults are published surface mapping; the scored faults are "
            "expert interpretations of the GeoDAWN geophysics. Compare policies, not leaderboards.",
            "The FP term is a global sum over the whole raster, exactly as the official metric - the "
            "truth set is sparse, so a broad, diffuse prediction is punished here too.",
            "Code-2 pixels are 'absent from the training labels', which is not the same as 'absent "
            "from the literature'.",
        ],
    }

    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\ntruth: {a.truth} -> {n_truth} px ({n_truth * 0.01:.1f} km of fault trace)")
    print(f"as submitted: DTI {res['as_submitted']['dti']:.4f}  "
          f"(TP_w {res['as_submitted']['TP_w']:.0f} FP_w {res['as_submitted']['FP_w']:.0f} "
          f"FN_w {res['as_submitted']['FN_w']:.0f})")
    for k, v in res["baselines"].items():
        print(f"  baseline {k:<28} DTI {v['dti']:.4f}")
    if "acceptance" in res:
        print(f"  acceptance catalogue-copy == 0: {'PASS' if res['acceptance']['passed'] else 'FAIL'}")
    print(f"wrote {op}")
    return 0 if res.get("acceptance", {}).get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
