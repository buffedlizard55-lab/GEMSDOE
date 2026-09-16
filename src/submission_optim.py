"""Metric-optimal submission shaping.

WHY THIS EXISTS - arithmetic, not opinion.  From the official metric
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric):

    TP_w = sum_{g in G} max_{x in B(g,R)} p(x) k(d(x,g))
    FP_w = sum_{x: p(x)>0} p(x) * (1 - max_{g in G} k(d(x,g)))
    DTI  = TP_w / (TP_w + 0.2 FP_w + 0.8 FN_w)

Two consequences follow directly:

1. FP_w is a SUM OVER AREA of probability mass, while |G| is only the fault pixels
   (~1.4% of the GeoDAWN grid in the public-source reconstruction).  A diffuse
   probability field (mean p = 0.2 over 326k px = 65k FP mass vs |G| = 4.6k) is
   therefore crushed: DTI <= TP_w / (0.2*FP_w) -> tiny.  Measured in this repo:
   mean-p 0.20 submission scored 0.0567 (see scripts/measure_submission_variants.py).

2. TP_w takes a MAX over the R-neighborhood of each ground-truth pixel.  So a single
   p=1 pixel within 3 px of a GT pixel earns ~all the credit that pixel can earn;
   the neighbouring 4-12 pixels of the same fault trace add nothing to TP_w but each add
   0.2*(1-k(d)) to FP_w.  Hence: keep probability on a THIN set that still dominates the
   predicted fault corridor, and zero it everywhere else.

   Marginal analysis (verified numerically in tests/test_metric.py): at pixel x, adding
   probability v (when it becomes the argmax for some g) changes DTI by
        v * (k(d(x,g)) * D - 0.2 * N) / D^2,   N = TP_w, D = TP_w + 0.2 FP_w + 0.8 FN_w
   so it pays iff  k > 0.2 * DTI.  At DTI = 0.6 that is k > 0.12, i.e. d < 2.64 px.
   Beyond R = 3 px it never pays unless the pixel really is GT.

The reference solution hints at the same thing in cell 17: "NOTE: it may be advantageous
to threshold this map for better scoring".  We generalise it to thinning + floor search.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from .metrics import kernel_offsets

__all__ = ["floor_sharpen", "dominant_thin", "dilate_mask", "optimize_submission",
           "search_threshold", "shaping_thresholds"]


def floor_sharpen(p: np.ndarray, t0: float = 0.0, gamma: float = 1.0, hard: bool = True) -> np.ndarray:
    """Remove background mass, then (optionally) push the remainder toward 1.

    t0: values <= t0 are set to exactly 0 (not merely scaled down) - the FP term sums
        probability, so residual background must be *zero*, not small-but-nonzero.
    gamma: >1 sharpens (only matters when hard=False).
    """
    p = np.nan_to_num(np.asarray(p, np.float32), nan=0.0)
    if hard:
        return (p > t0).astype(np.float32)
    q = np.clip((p - t0) / max(1e-6, 1.0 - t0), 0.0, 1.0) ** gamma
    q[p <= t0] = 0.0
    return q.astype(np.float32)


def dominant_thin(mask: np.ndarray, R: int = 3, p: np.ndarray | None = None,
                  max_extra: int = 40) -> np.ndarray:
    """Reduce a binary fault mask to the smallest subset that still keeps every original
    mask pixel within R of a kept pixel (a distance-R dominating set), so that TP credit
    is preserved while FP mass collapses.

    Order of operations: skeletonize (topology-preserving thinning) -> greedy fill for any
    pixels the skeleton failed to dominate (wide blobs), scored by probability mass covered.
    """
    m = mask > 0.5
    if not m.any():
        return m.astype(np.float32)
    skel = m.copy()
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(m)
        if sk.any():
            skel = sk
    except Exception:                                   # pragma: no cover
        pass
    # keep only skeleton pixels inside the mask (skeletonize stays inside by construction)
    sel = skel & m
    if sel.any():
        # uncovered = mask pixels farther than R from any selected pixel
        for _ in range(max_extra):
            d = distance_transform_edt(~sel)
            uncovered = m & (d > R)
            if not uncovered.any():
                break
            # add the pixel covering the most uncovered ones (within R), tie-break by p
            from scipy.ndimage import uniform_filter
            dens = uniform_filter(uncovered.astype(np.float64), size=2 * R + 1)
            cand = m & ~sel
            if p is not None:
                score = dens + 1e-3 * p
            else:
                score = dens
            score = np.where(cand, score, -np.inf)
            iy, ix = np.unravel_index(np.argmax(score), score.shape)
            sel[iy, ix] = True
    return sel.astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Grow a kept set by `radius` pixels (disk).

    WHY A KNOB AND NOT ALWAYS THIN: thinning is optimal only if the predicted trace sits on
    the true trace.  TP_w takes a max within R = 3 px, so a prediction that is off by up to
    R still scores as if it were exact -- but a prediction that is off by MORE than R scores
    nothing at all while still paying 0.2 per unit of FP mass.  Against the *catalogued*
    faults the model localises well (it trained on them), so thin wins there.  Against the
    *scored* faults -- new faults absent from the catalogue (rules SS1.1/SS3.5) -- the model
    has never seen them and its localisation error is strictly larger.  Independent evidence
    that it is at least one cell: Hermant et al. (2025) find expert-mapped Quaternary fault
    traces and the USGS compilation differ by up to 400 m, i.e. 4 cells at 100 m.
    Growing the kept set by k pixels buys k cells of tolerance for a bounded FP cost; the
    right k is an empirical question, which is what `dilate` is for.  See
    scripts/measure_shift_robustness.py for the measurement.
    """
    m = np.asarray(mask) > 0.5
    if radius <= 0 or not m.any():
        return m.astype(np.float32)
    try:
        from skimage.morphology import disk
        st = disk(int(radius))
    except Exception:                                   # pragma: no cover
        st = None
    return binary_dilation(m, structure=st, iterations=1 if st is not None else int(radius)
                           ).astype(np.float32)


def optimize_submission(p: np.ndarray, R: int = 3, t0: float = 0.3, thin: bool = True,
                        hard: bool = True, gamma: float = 1.0, dilate: int = 0) -> np.ndarray:
    """Full shaping pipeline -> probability field ready to write as a submission.

    `dilate` (pixels, default 0 = pure skeleton) widens the kept set after thinning; see
    dilate_mask for why the optimum against the scored universe may not be 0.
    """
    q = floor_sharpen(p, t0=t0, gamma=gamma, hard=hard)
    if thin:
        q = dominant_thin(q, R=R, p=p)
    if dilate:
        q = dilate_mask(q, radius=int(dilate))
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def shaping_thresholds(n: int = 15, lo: float = 1e-4, hi: float = 0.9) -> np.ndarray:
    """Floor candidates for the shaping search, log-spaced from `lo` to `hi`.

    WHY NOT linspace(0.02, 0.9): the full-raster smoke run (Actions run 34876843912)
    selected t0 = 0.02 -- the *first* point of the old grid.  An optimum sitting exactly on
    a grid boundary means the grid was the constraint, not the data: an under-trained model
    emits a low-contrast probability field whose useful separation lives well below 0.02,
    so the search could not reach it and the reported optimum was an artefact of the
    endpoint.  Log spacing also matches how the floor behaves -- what matters is the order
    of magnitude of the mass being cut, and DTI changes far faster between 0.001 and 0.01
    than between 0.5 and 0.51.

    Always includes 0.0 (no floor) as an explicit reference row.
    """
    n = max(2, int(n))
    grid = np.geomspace(max(1e-12, float(lo)), float(hi), n)
    return np.unique(np.concatenate(([0.0], grid)))


def search_threshold(p: np.ndarray, gt: np.ndarray, R: int = 3,
                     thresholds=None, thin_options=(False, True), dilate_options=(0,)):
    """Grid search floor + thinning (+ optional dilation) on HELD-OUT validation
    predictions (never on test).

    Returns (best_params, best_dti, table).  The table is what makes this auditable:
    it shows the metric's shape, so the chosen point is reviewable, not magic.
    """
    from .metrics import compute_distance_weighted_tversky
    if thresholds is None:
        thresholds = shaping_thresholds(45)
    table = [(float("nan"), False, 0, float(compute_distance_weighted_tversky(p, gt, R_pixels=R)),
              float(np.nansum(p)))]                      # row 0 = unshaped reference point
    best = (-1.0, None)
    for t in thresholds:
        for thin in thin_options:
            for dila in dilate_options:
                q = optimize_submission(p, R=R, t0=float(t), thin=thin, dilate=int(dila))
                d = compute_distance_weighted_tversky(q, gt, R_pixels=R)
                table.append((float(t), bool(thin), int(dila), float(d), float(q.sum())))
                if d > best[0]:
                    best = (d, dict(t0=float(t), thin=bool(thin), dilate=int(dila)))
    return best[1], best[0], table
