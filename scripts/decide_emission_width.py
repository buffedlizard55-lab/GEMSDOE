#!/usr/bin/env python3
"""Decide the submission-shaping policy in public, from measurements that disagree.

THE QUESTION
------------
``src/submission_optim.optimize_submission`` shapes the ensemble's probability field into a
submission with three knobs: a floor ``t0``, distance-R thinning (keep one skeleton pixel per
R-neighbourhood), and an emission width ``dilate`` that grows the kept set by k pixels.  The
shipped default is ``t0 = 0.470``, ``thin = True``, ``dilate = 0`` (a ~21,500-pixel skeleton).

Three measurements of the width question exist in this repository, and they point in opposite
directions:

  A. in-domain held-out crops (truth = the catalogue the model trained on)
       skeleton 0.1903  >  1 px 0.1525  >  2 px 0.1281  >  3 px 0.1128  >  6 px 0.0908
     -> narrower wins, decisively.

  B. shifted-label stress test (truth = the same catalogue translated 3-4 px)
       0.0555 -> 0.0991 (3 px) -> 0.1190 (6 px)
     -> wider wins, decisively.

  C. proxy-only population (truth = 6,166 km of USGS SGMC fault trace that the training labels
     do NOT contain; the closest measurable stand-in for the scored "new fault" universe)
       skeleton 0.0144  <  1 px 0.0205  <  2 px 0.0256  <  3 px 0.0312  <  4 px 0.0340  <  6 px 0.0395
     -> wider wins, monotonically.

The disagreement is not noise, it is the point.  A and C measure the SAME operator against
populations that differ in exactly the way the competition cares about: in A the model has seen
the truth, so its trace sits on the label and widening only adds false-positive mass; in C the
truth is a fault population the labels lack, so the trace can be off by more than the metric's
R = 3-pixel tolerance and a wider band converts misses into hits.

WHAT THIS SCRIPT ADDS
---------------------
Absolute DTIs are not comparable across populations, and the scored set's size |G| is unknown,
so "which policy is better" cannot be read off any single number.  But the metric's two error
terms scale differently with |G|: the prediction fixes the wrong-mass F = FP_w, while the missing
mass beta*(|G| - TP_w) grows with the hidden truth.  Writing c = TP_w/|G_proxy| for the measured
coverage fraction and holding it fixed, a policy measured on the proxy population has

    DTI(|G|) = c|G| / ( alpha*(c|G| + F) + beta*|G| )          [exact at |G| = |G_proxy|]

(algebraically identical to the official TP/(TP + alpha*FP + beta*FN) because alpha + beta = 1 --
that identity is asserted below, since the whole projection rests on it).  Two consequences:

  * every policy has an asymptote c/(alpha*c + beta) as |G| grows: a policy that emits more
    coverage wins in the limit no matter how much wrong mass it carries, because wrong mass is
    discounted by alpha = 0.2 and never scales with |G|;
  * for any two policies there is a single crossover size G* above which the higher-coverage
    policy wins, and below which the cleaner one wins.

That makes the decision a *sensitivity* question with a computable answer: which of the plausible
scored-truth sizes are we actually in?  This script reports the crossovers, projects every
measured policy onto a range of plausible |G|, and states the verdict against the repository's
pre-registered rule (a change must beat the current default by > 0.01 on the new-fault-like
population AND dominate it across plausible truth sizes AND reproduce on a second ensemble).

THE FINDING THAT MATTERS MORE THAN THE WIDTH
--------------------------------------------
The sweep measured a constant-ones submission (fill the data footprint with 1.0) on population C:
DTI 0.0585, which BEATS the shipped skeleton's 0.0247 and every swept shaping candidate.  A
constant map has no skill at all; it wins because recall is worth four times precision under this
metric (alpha = 0.2, beta = 0.8) and because it pays its false-positive mass in a currency the
denominator discounts.  Any policy that emits less coverage than a constant map is therefore not
"conservative", it is *under-emitting*, and the crossover says at what truth size that becomes
true.  This is reported here because it is actionable and it is uncomfortable.

NOT A LEADERBOARD PREDICTION.  Population C is state-geological-survey surface mapping, not the
expert interpretation of GeoDAWN geophysics the prize scores; the numbers are for comparing
policies, and the projected |G| values are assumptions printed as assumptions.

USAGE
    python scripts/decide_emission_width.py --out data/evidence/emission_decision.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import json
from pathlib import Path

ALPHA, BETA = 0.2, 0.8
PX_KM = 0.1            # 100 m pixels
AOI_AREA_KM2 = 3292 * 3730 * PX_KM ** 2          # 122,791.6 km^2
GEODAWN_AREA_KM2 = 51857.0                       # USGS GeoDAWN data release, 10.5066/P93LGLVQ


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def provided(res: dict) -> dict:
    """The score of the raster handed to --pred, under either spelling.

    Evidence written after 2026-09-16 calls it `as_provided` (the old name, `as_submitted`, called
    every --pred raster a submission, which was wrong in the shaping sweep); earlier evidence files
    still carry the old key, so read both.
    """
    return res.get("as_provided") or res["as_submitted"]


def policy_row(name: str, truth_px: int, res: dict, note: str = "") -> dict:
    """Turn a measured score into the two parameters that control its scaling."""
    tp, fp = float(res["TP_w"]), float(res["FP_w"])
    return {"policy": name, "measured_dti": round(float(res["dti"]), 6),
            "coverage_fraction": round(tp / float(truth_px), 6), "wrong_mass_FP_w": round(fp, 3),
            "emission_px": int(res.get("emission_px") or 0), "note": note}


def project(row: dict, truth_px: float) -> float:
    """DTI for a hypothetical scored truth of `truth_px`, from a measured (coverage, FP) pair."""
    c, f = row["coverage_fraction"], row["wrong_mass_FP_w"]
    tp = c * float(truth_px)
    return float(tp / (ALPHA * (tp + f) + BETA * float(truth_px) + 1e-12))


def crossover(p1: dict, p2: dict) -> dict:
    """Truth size at which two policies' projected curves cross (or None if they do not).

    Solving c1|G|/(a(c1|G|+F1)+b|G|) = c2|G|/(a(c2|G|+F2)+b|G|) for |G| gives
        G* = a*(c2*F1 - c1*F2) / (b*(c1 - c2))
    A positive G* means the lower-coverage policy is better below it and the higher-coverage one
    above it (when c1 < c2); a negative or undefined G* means one policy dominates everywhere.
    """
    a, b = ALPHA, BETA
    c1, f1 = p1["coverage_fraction"], p1["wrong_mass_FP_w"]
    c2, f2 = p2["coverage_fraction"], p2["wrong_mass_FP_w"]
    if abs(c1 - c2) < 1e-12:
        return {"between": [p1["policy"], p2["policy"]], "crossover_px": None,
                "reason": "identical coverage: the policy with less wrong mass wins everywhere"}
    g = a * (c2 * f1 - c1 * f2) / (b * (c1 - c2))
    out = {"between": [p1["policy"], p2["policy"]], "crossover_px": round(float(g), 1)}
    if g <= 0:
        winner = p1["policy"] if c1 > c2 else p2["policy"]
        out["reason"] = f"no crossing at a positive truth size: {winner} wins at every size"
    else:
        wider = p2["policy"] if c2 > c1 else p1["policy"]
        cleaner = p1["policy"] if c2 > c1 else p2["policy"]
        out["reason"] = (f"{cleaner} wins below {round(float(g), 1)} px "
                         f"({round(float(g) * PX_KM, 1)} km); {wider} wins above")
    return out


def width_gain_table(sweep: dict) -> dict:
    """Per-floor DTI contrast between the widest swept band and the pure skeleton.

    This is the FLOOR-CONTROLLED form of the width question, and the reason it is computed inside
    one sweep file rather than across sweeps: absolute proxy DTI differs between ensembles (each
    fold set produces a different probability field), so comparing ensemble 2's number with
    ensemble 1's number would confound the width with the model.  For every swept floor ``t0``
    that has both ``dilate=0`` and the widest ``dilate``, the contrast is taken at the SAME floor,
    so the sign and the size of the gain are attributable to the emission width alone.

    The repository's pre-registered acceptance rule for the width change is "same sign and
    > 0.01" - evaluated here as: every swept floor gains, and the mean gain exceeds 0.01.
    """
    # HARD rows only: a ramp candidate has the same support as the hard band of its width but
    # different values, so it is a different policy - and, critically, its presence in this table
    # would define "the widest band" by a row that is not a band width at all.
    rows = [r for r in sweep["results"]["shaping_sweep"] if r["thin"] and not r.get("soft")]
    widths = sorted({int(r["dilate"]) for r in rows})
    if not widths:
        return dict(widest_px=None, per_floor=[], n_floors=0, mean_gain=None, min_gain=None)
    widest = max(widths)
    per_floor = []
    for t in sorted({float(r["t0"]) for r in rows}):
        d0 = next((float(r["dti"]) for r in rows
                   if float(r["t0"]) == t and int(r["dilate"]) == 0), None)
        dw = next((float(r["dti"]) for r in rows
                   if float(r["t0"]) == t and int(r["dilate"]) == widest), None)
        if d0 is None or dw is None:
            continue
        per_floor.append(dict(t0=t, dti_width0=round(d0, 6), dti_widest=round(dw, 6),
                              gain=round(dw - d0, 6)))
    gains = [p["gain"] for p in per_floor]
    return dict(widest_px=widest, per_floor=per_floor, n_floors=len(gains),
                mean_gain=(round(sum(gains) / len(gains), 6) if gains else None),
                min_gain=(min(gains) if gains else None),
                max_gain=(max(gains) if gains else None),
                all_floors_positive=bool(gains) and all(g > 0 for g in gains))


def policy_reproduction(sweep: dict, t0: float, width: int) -> dict | None:
    """Look up the SAME hard policy in another ensemble's sweep and contrast it with THAT sweep's
    own reference policy.

    Absolute proxy DTI is not comparable across ensembles - each fold set produces a different
    probability field - so a candidate is only "reproduced" if the contrast against the reference
    policy of the same sweep keeps its sign and size.  This is the floor-change counterpart of
    ``width_gain_table``: the width table controls the floor and varies the width, this controls
    nothing and varies the whole policy, which is what a candidate that changes BOTH needs.
    """
    rows = [r for r in sweep["results"]["shaping_sweep"] if r["thin"] and not r.get("soft")]
    target = next((r for r in rows if float(r["t0"]) == float(t0) and int(r["dilate"]) == int(width)),
                  None)
    ref = (sweep["results"].get("sweep_verdict") or {}).get("current_policy_dti")
    if target is None or ref is None:
        return None
    return {"t0": float(t0), "width_px": int(width),
            "measured_dti": round(float(target["dti"]), 6),
            "reference_policy_dti": round(float(ref), 6),
            "contrast_vs_reference": round(float(target["dti"]) - float(ref), 6),
            "reference_is": "sweep_verdict.current_policy_dti of that same sweep"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-eval", default="data/evidence/proxy/eval_submission.json")
    ap.add_argument("--reblend-eval", default="data/evidence/proxy/eval_reblend_submission.json")
    ap.add_argument("--sweep", default="data/evidence/proxy/eval_sweep.json")
    ap.add_argument("--miss-distance", default="data/evidence/proxy/miss_distance-ensemble1.json",
                    help="scripts/measure_miss_distance.py evidence: the exact metric as a function "
                         "of the emitted band width, plus the localization/detection split")
    ap.add_argument("--second-sweep", default=None,
                    help="sweep of a SECOND, independently trained ensemble (same recipe, different "
                         "seed/folds). Evaluated as condition 3: the width gain must keep its sign "
                         "and exceed 0.01 with every floor controlled. Written by the proxy-eval "
                         "workflow as data/evidence/proxy/eval_sweep-<label>.json")
    ap.add_argument("--in-domain",
                    default="data/evidence/runs/35042805806-experiment/blend_report.json")
    ap.add_argument("--out", default="data/evidence/emission_decision.json")
    a = ap.parse_args()

    assert abs((ALPHA + BETA) - 1.0) < 1e-12, "the projection identity requires alpha + beta == 1"

    # ------------------------------------------------------------------ measured policies (C)
    rows: list[dict] = []
    ev = load(Path(a.proxy_eval))
    gp = int(ev["truth"]["px"])
    rows.append(policy_row("shipped_skeleton_dilate0", gp, provided(ev["results"]),
                           "the committed submission: t0=0.470, thin, width 0"))
    blanks = ev["results"]["baselines"]
    rows.append(policy_row("baseline_blanket_ones", gp, blanks["blanket_ones"],
                           "fill the data footprint with 1.0 - no skill, maximum recall"))
    rows.append(policy_row("baseline_catalogue_plus_submission", gp,
                           blanks["catalogue_plus_submission"], "the catalogue unioned with ours"))
    if "blanket_ones_whole_grid" in blanks:
        rows.append(policy_row("baseline_blanket_ones_whole_grid", gp,
                               blanks["blanket_ones_whole_grid"],
                               "illegal submission (must be NaN outside the footprint), "
                               "recorded for scale"))
    rb = load(Path(a.reblend_eval))
    rows.append(policy_row("reblend_hard_39517px", gp, provided(rb["results"]),
                           "the fold pair re-blended without the pre-shaping floor"))
    sw = load(Path(a.sweep))
    swept: list[tuple[float, int, dict]] = []
    ramp: list[dict] = []
    for row in sw["results"]["shaping_sweep"]:
        if not row["thin"]:
            continue
        # HARD candidates only in the width machinery below: a ramp candidate has its own support
        # and its own emitted values, so f"width {d} px" does not identify it, and mixing the two
        # would put a policy into the width curve that the width curve cannot describe.
        if row.get("soft"):
            ramp.append(row)
            continue
        swept.append((float(row["t0"]), int(row["dilate"]), row))
        if row["dilate"] not in (0, 3, 6) or row["t0"] not in (0.0, 0.0432675):
            continue
        rows.append(policy_row(f"sweep_t0_{row['t0']:g}_width{row['dilate']}px", gp, row,
                               f"floor {row['t0']:g}, thinning on, width {row['dilate']} px"))
    if not swept:
        raise SystemExit("the sweep evidence carries no thinned rows - nothing to decide from")
    # The widest candidate actually measured, whatever the grid was: the workflow's dilate grid is
    # an input, so hard-coding "6 px" here would silently compare against nothing when it changes.
    widest = max(d for _, d, _ in swept)
    # Among the widest band, the floor that scores best on this population is the fair wide
    # candidate (picking the highest floor would pick the collapse case, which scores ~0).
    # The floor axis is a candidate axis in its own right.  The session-12 extended grid showed the
    # best HARD candidate can be a floor change rather than a width change, and a ranking that
    # admitted only floors {0, 0.0433} would then decide a width question while silently excluding
    # what the search actually found.  Every swept floor contributes its single best width.
    best_per_floor: dict[float, dict] = {}
    for t, d, r in swept:
        if t not in best_per_floor or float(r["dti"]) > float(best_per_floor[t]["dti"]):
            best_per_floor[t] = r
    existing_names = {r["policy"] for r in rows}
    for t, r in sorted(best_per_floor.items()):
        name = f"sweep_best_t0_{t:g}_width{int(r['dilate'])}px"
        if name in existing_names:
            continue
        rows.append(policy_row(name, gp, r, f"best hard candidate at floor {t:g} "
                                            f"(thinning on): width {int(r['dilate'])} px"))
    wide_t0, _, wide_row = max([(t, d, r) for t, d, r in swept if d == widest],
                               key=lambda x: (x[2]["dti"], -x[0]))
    wide_name = f"sweep_t0_{wide_t0:g}_width{widest}px"
    if wide_name not in {r["policy"] for r in rows}:
        rows.append(policy_row(wide_name, gp, wide_row,
                               f"floor {wide_t0:g}, thinning on, width {widest} px (widest swept)"))
    print(f"widest swept emission: {widest} px at floor {wide_t0:g}")
    rows.append(policy_row("ensemble_soft_map", gp, provided(sw["results"]),
                           "the raw pre-shaping ensemble mean (not a legal submission: values "
                           "outside the footprint are not NaN)"))
    # Ramp emission (session 11): identical support to the hard band of the same width, values
    # decaying to 0 at the band edge.  The best ramp per width is a policy in the same ranking as
    # the hard candidates, so a decision that ignores it would be a decision made on a subset.
    ramp_best: dict[int, dict] = {}
    for row in ramp:
        d = int(row["dilate"])
        if d not in ramp_best or float(row["dti"]) > float(ramp_best[d]["dti"]):
            ramp_best[d] = row
    for d, row in sorted(ramp_best.items()):
        rows.append(policy_row(
            f"sweep_ramp_t0_{float(row['t0']):g}_width{d}px_gamma{float(row.get('gamma', 1.0)):g}",
            gp, row,
            f"ramp values (gamma {float(row.get('gamma', 1.0)):g}) on the width-{d} px support, "
            f"floor {float(row['t0']):g}, thinning on"))
    if ramp:
        print(f"ramp emission candidates scored: {len(ramp)}; best per width: "
              f"{ {d: round(float(r['dti']), 6) for d, r in sorted(ramp_best.items())} }")

    # ------------------------------------------------------------------ plausible |G|
    # Two anchors, both stated as assumptions.  1) Density scaling: the proxy catalogue has a
    # measured fault density over this AOI; the scored faults are drawn from the GeoDAWN blocks,
    # which cover a known area of it.  2) The catalogue itself (labels.tif) is the other known
    # population inside the same AOI.
    density_km_per_km2 = (gp * PX_KM) / AOI_AREA_KM2
    plausible = {
        "assumption_density_km_per_km2": round(density_km_per_km2, 5),
        "assumption_geodawn_area_km2": GEODAWN_AREA_KM2,
        "assumption_aoi_area_km2": round(AOI_AREA_KM2, 1),
        "anchors": [
            {"name": "same density over the GeoDAWN blocks",
             "truth_px": int(round(density_km_per_km2 * GEODAWN_AREA_KM2 / PX_KM)),
             "why": "the scored faults come from the blocks the geophysics was flown over"},
            {"name": "same density over the whole AOI",
             "truth_px": gp, "why": "the measured population itself"},
            {"name": "the public catalogue (labels.tif) alone",
             "truth_px": 60988, "why": "known: 6,099 km, the faults the model trained on"},
        ],
    }
    sizes = sorted({500, 1000, 2500, 5000, 10000, 20000, 30000, 50000, 61664, 100000, 250000}
                   | {x["truth_px"] for x in plausible["anchors"]})

    # ------------------------------------------------------------------ in-domain (A), for contrast
    idr = load(Path(a.in_domain))
    ct = idr["shaping"]["calibration_table"]
    in_domain = [
        {"t0": r["t0"], "thin": r["thin"], "dilate": r["dilate"], "mean_heldout_dti": r["mean_dti"]}
        for r in ct if r.get("t0") in (0.0, 0.469674) and r.get("thin") and
        r.get("dilate") in (0, 1, 3, 6)]
    in_domain_rows = sorted(in_domain, key=lambda r: (r["t0"], r["dilate"]))

    # ------------------------------------------------------------------ verdict
    shipped = next(r for r in rows if r["policy"] == "shipped_skeleton_dilate0")
    wide = next(r for r in rows if r["policy"] == wide_name)
    blanket = next(r for r in rows if r["policy"] == "baseline_blanket_ones")
    v = {
        "pre_registered_rule": sw["results"]["sweep_verdict"]["acceptance_rule"],
        "proxy_population_ranking": [r["policy"] for r in
                                     sorted(rows, key=lambda r: -r["measured_dti"])],
        "wide_vs_shipped": crossover(shipped, wide),
        "blanket_vs_shipped": crossover(shipped, blanket),
        "blanket_vs_wide": crossover(wide, blanket),
        "in_domain_contrast": ("on the population the model trained on, the same widening costs "
                               "0.1903 -> 0.0908 (skeleton -> 6 px): the sign of the effect flips "
                               "with the population, which is why neither number alone can decide it"),
        "conditions": [],
        "conclusion": "",
        "top_priority": ("beating the constant-ones baseline on the new-fault-like population is "
                         "worth more than the width choice: the shipped skeleton does not beat it at "
                         "any truth size above the crossover printed above, and the leaderboard's "
                         "top score (0.1972) is 3.4x the blanket's 0.0585 on this population"),
    }
    # Sweep policies only: the baselines (blanket, catalogue copy) are ranked in the same table but
    # they are controls, not candidates the shaping pipeline can produce.
    best_cand = max((r for r in rows if r["policy"].startswith("sweep_")),
                    key=lambda r: r["measured_dti"])
    v["best_measured_candidate"] = {
        "policy": best_cand["policy"],
        "measured_dti": best_cand["measured_dti"],
        "note": best_cand["note"],
        "contrast_vs_shipped": round(best_cand["measured_dti"] - shipped["measured_dti"], 6),
        "crosses_shipped": crossover(shipped, best_cand),
        "reproduced_on_second_ensemble": None,
        "why_this_is_recorded": ("the pre-registered rule's candidate set has to be stated, not "
                                 "assumed: the best measured candidate on this population is "
                                 "ranked here, so a verdict cannot be reached with it excluded"),
    }
    m_best = re.match(r"sweep_best_t0_([0-9.]+)_width([0-9]+)px$", best_cand["policy"]) or \
        re.match(r"sweep_t0_([0-9.]+)_width([0-9]+)px$", best_cand["policy"])
    if m_best:
        cand_floor = float(m_best.group(1))
        # The floor and the width are alternative policies, not additive knobs: at the best floor the
        # best width may be 0 px.  Recording the whole width curve of the winning floor makes that
        # visible in the record instead of leaving it to a reader to reconstruct from the sweep file.
        v["best_measured_candidate"]["widths_at_that_floor"] = {
            str(int(r["dilate"])): round(float(r["dti"]), 6)
            for t, d, r in sorted(swept, key=lambda x: x[1]) if t == cand_floor}
        w_at = v["best_measured_candidate"]["widths_at_that_floor"]
        v["best_measured_candidate"]["width_optimum_at_that_floor_px"] = (
            int(max(w_at, key=lambda k: w_at[k])) if w_at else None)
    if a.second_sweep and Path(a.second_sweep).exists():
        m2 = m_best
        if m2:
            v["best_measured_candidate"]["reproduced_on_second_ensemble"] = policy_reproduction(
                load(Path(a.second_sweep)), float(m2.group(1)), int(m2.group(2)))
    gain = round(wide["measured_dti"] - shipped["measured_dti"], 6)
    v["conditions"].append({
        "condition": "beats the shipped default on the new-fault-like population by > 0.01",
        "required": "> 0.01 measured proxy DTI", "measured": gain, "passes": bool(gain > 0.01)})
    cw = v["wide_vs_shipped"]["crossover_px"]
    if cw is None or cw <= 0:
        v["conditions"].append({"condition": "dominates the shipped default across plausible "
                                             "scored-truth sizes", "required": True,
                                "measured": "no crossing: it wins everywhere", "passes": True})
        v["conclusion"] = ("no width change needed: the wider candidate is not worse anywhere, so "
                           "the shipped skeleton can stay until something better is measured")
    else:
        anchor_rows = [(x["name"], project(shipped, x["truth_px"]), project(wide, x["truth_px"]))
                       for x in plausible["anchors"]]
        wins = sum(1 for _, s_, w in anchor_rows if w > s_)
        v["anchor_projection"] = [
            {"anchor": n, "truth_px": int(next(x["truth_px"] for x in plausible["anchors"]
                                               if x["name"] == n)),
             "shipped_dti": round(s_, 4), "wide_dti": round(w, 4), "wide_wins": bool(w > s_)}
            for n, s_, w in anchor_rows]
        v["conditions"].append({
            "condition": "wins at the plausible scored-truth sizes (not only in the large-|G| limit)",
            "required": "every anchor favours the wider band",
            "measured": f"{wins}/{len(anchor_rows)} anchors favour it; crossover {cw:,.0f} px "
                        f"({cw * PX_KM:,.0f} km)",
            "passes": bool(wins == len(anchor_rows))})
        first_gain = width_gain_table(sw)
        second_gain = None
        if a.second_sweep and Path(a.second_sweep).exists():
            second_gain = width_gain_table(load(Path(a.second_sweep)))
        reproduced = bool(second_gain and second_gain["n_floors"] and
                          second_gain["all_floors_positive"] and
                          (second_gain["mean_gain"] or 0.0) > 0.01)
        v["conditions"].append({
            "condition": "reproduced on a second, independently trained ensemble",
            "required": "same sign and > 0.01 on the second ensemble",
            "measured": (
                f"second ensemble ({a.second_sweep}): widest {second_gain['widest_px']} px vs 0 px "
                f"at {second_gain['n_floors']} common floors -> mean gain "
                f"{second_gain['mean_gain']:+.4f}, min {second_gain['min_gain']:+.4f}, "
                f"all floors {'positive' if second_gain['all_floors_positive'] else 'NOT all positive'}"
                if second_gain and second_gain["n_floors"] else
                "NOT MEASURED: needs a second, independently trained ensemble sweep "
                "(see SUGGESTIONS.md 7.1)"),
            "passes": reproduced})
        v["width_gain_first_ensemble"] = first_gain
        if second_gain is not None:
            v["width_gain_second_ensemble"] = second_gain
        if reproduced:
            v["conclusion"] = (
                f"WIDEN - all three pre-registered conditions are met. The {widest}-px band beats the "
                f"shipped skeleton on the new-fault-like population by {gain:+.4f} (condition 1), at "
                f"{wins}/{len(anchor_rows)} plausible scored-truth anchors (condition 2), and the "
                f"floor-controlled gain reproduces on a second, independently trained ensemble "
                f"({second_gain['mean_gain']:+.4f} mean over {second_gain['n_floors']} floors, "
                f"condition 3). The shipped default is therefore changed to emit the measured band: "
                f"see the workflow's --min-dilate option and SUGGESTIONS.md")
        else:
            v["conclusion"] = (
                f"widen, but not yet: the {widest}-px band beats the shipped skeleton on the "
                f"new-fault-like population by {gain:+.4f} (condition 1) and at "
                f"{wins}/{len(anchor_rows)} plausible scored-truth anchors (condition 2), while a "
                f"scored truth below {cw:,.0f} px ({cw * PX_KM:,.0f} km) would still favour the "
                f"narrower skeleton. Condition 3 - reproduction on a second ensemble - is unmet and "
                f"is the blocking item, so the shipped default is unchanged and this file is the "
                f"record of why")

    # ------------------------------------------------------------------ measured width curve
    # The sweep searches floor x width on the ensemble probability map; this is the independent,
    # floor-free measurement on the SHIPPED hard submission (scripts/measure_miss_distance.py):
    # dilate the emitted set by k px and score with the same metric.  It answers what the sweep
    # could not - how far the misses actually are - and gives the projected optimum width per
    # assumed scored-truth size.
    md = None
    if a.miss_distance and Path(a.miss_distance).exists():
        m = load(Path(a.miss_distance))
        md = {"source": a.miss_distance,
              "truth_km": m.get("truth_km"), "emitted_km": m.get("emitted_km"),
              "miss_distance_percentiles_px": m["miss_geometry"]["percentiles_px"],
              "detection_failure_share_beyond_12px": m["verdict"]["detection_failure_share"],
              "best_width_px": m["verdict"]["best_width_px"],
              "gain_from_widening": m["verdict"]["gain_from_widening"],
              "widening_starts_winning_above_px": m.get("widening_starts_winning_above_px"),
              "projection": m.get("projection_over_scored_truth_size")}

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/decide_emission_width.py",
        "purpose": ("reconcile the three disagreeing emission-width measurements and decide the "
                    "shipped shaping policy on the record, using the metric's own scaling in |G|"),
        "inputs": {"proxy_eval": a.proxy_eval, "reblend_eval": a.reblend_eval, "sweep": a.sweep,
                   "second_sweep": a.second_sweep, "miss_distance": a.miss_distance,
                   "in_domain": a.in_domain,
                   "metric": {"alpha": ALPHA, "beta": BETA, "R_pixels": 3, "R_meters": 300}},
        "proxy_truth_px": gp,
        "policies": rows,
        "width_vs_scored_truth_size": md,
        "crossovers": {"wide_vs_shipped": v["wide_vs_shipped"],
                       "widest_swept_px": widest,
                       "blanket_vs_shipped": v["blanket_vs_shipped"],
                       "blanket_vs_wide": v["blanket_vs_wide"]},
        "projection_table": [
            {"truth_px": int(g), "truth_km": round(g * PX_KM, 1),
             **{r["policy"]: round(project(r, g), 5) for r in rows}} for g in sizes],
        "plausible_scored_truth": plausible,
        "in_domain_measurement": {"rows": in_domain_rows,
                                  "measures": "held-out crops of the public catalogue",
                                  "caveat": "upper bound on plumbing, not on skill"},
        "verdict": v,
        "caveats": [
            "The projection holds coverage and wrong mass fixed while |G| changes; a model that "
            "would be retrained for a different-sized truth would not hold either.",
            "Population C is USGS state-map surface mapping, not the expert interpretation of "
            "GeoDAWN geophysics that the prize scores. Policy comparisons only.",
            "The scored truth size is unknowable from here; the anchors are assumptions printed "
            "as assumptions, not estimates with confidence intervals.",
        ],
    }
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"proxy truth |G_proxy| = {gp:,} px ({gp * PX_KM:,.0f} km)\n")
    print("measured policies on the new-fault-like population (population C):")
    for r in sorted(rows, key=lambda r: -r["measured_dti"]):
        print("  %-34s DTI %.4f  coverage %6.2f%%  wrong mass FP_w %12.1f  emitted %9d px"
              % (r["policy"], r["measured_dti"], 100 * r["coverage_fraction"],
                 r["wrong_mass_FP_w"], r["emission_px"]))
    print("\ncrossovers (the truth size at which the wider/higher-coverage policy takes over):")
    for k, c in out["crossovers"].items():
        # `crossovers` also carries scalar context (`widest_swept_px`); subscripting every value
        # as a dict crashed this print AFTER the JSON was written (found 2026-09-17: the step
        # failed while its evidence landed, so a red step looked like a completed reconciliation).
        print("  %-22s %s" % (k, c["reason"] if isinstance(c, dict) else c))
    print("\nprojected DTI at the plausible anchors:")
    for x in v.get("anchor_projection", []):
        print("  %-40s shipped %.4f  wide-6px %.4f  %s"
              % (x["anchor"], x["shipped_dti"], x["wide_dti"],
                 "WIDE WINS" if x["wide_wins"] else "shipped wins"))
    bc = v["best_measured_candidate"]
    print("\nbest measured candidate on this population: %s" % bc["policy"])
    print("  DTI %.4f (%+.4f vs the shipped policy); %s"
          % (bc["measured_dti"], bc["contrast_vs_shipped"], bc["note"]))
    if bc["reproduced_on_second_ensemble"]:
        r2 = bc["reproduced_on_second_ensemble"]
        print("  second ensemble: DTI %.4f vs its own reference %.4f -> %+.4f"
              % (r2["measured_dti"], r2["reference_policy_dti"], r2["contrast_vs_reference"]))
    print("\nconclusion: %s" % v["conclusion"])
    print("top priority: %s" % v["top_priority"])
    print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
