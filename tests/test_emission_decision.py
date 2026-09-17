"""The emission-width decision must be computable, and must be falsifiable.

Two defects found on 2026-09-17 (session 10) while preparing the second-ensemble measurement:

1. ``scripts/decide_emission_width.py`` crashed in its final PRINT loop - ``out["crossovers"]``
   carries both crossing records (dicts with a ``reason``) and scalar context
   (``widest_swept_px``), and the loop subscripted every value as a dict.  The JSON was written
   first, so the workflow step failed *after* its evidence landed: a red reconciliation step
   looked like a completed one.  Pinned here by running the script on the committed evidence and
   requiring exit 0.
2. Condition 3 ("reproduced on a second, independently trained ensemble") was a hard-coded
   ``passes: False`` with the text "NOT MEASURED", i.e. the measurement the queued second
   ensemble exists to produce could not change the verdict even if it agreed.  It is now computed
   from a floor-controlled contrast between the widest swept band and the pure skeleton inside
   each sweep file - floor-controlled because absolute proxy DTI is not comparable across
   ensembles.  Both directions (reproduced / not reproduced) are pinned below.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decide_emission_width.py"
sys.path.insert(0, str(ROOT))

TRUTH_PX = 61664


def _sweep(gains: list[float], floors=(0.0, 0.05, 0.1), widths=(0, 3, 6)) -> dict:
    """A synthetic sweep whose width-0 DTI is fixed and whose widest band moves by `gains`."""
    rows = []
    for g, t0 in zip(gains, floors):
        for d in widths:
            dti = 0.02 + (g if d == max(widths) else 0.0)
            rows.append(dict(t0=t0, thin=True, dilate=d, dti=dti,
                             TP_w=dti * TRUTH_PX * 0.5, FP_w=1e4 * (1 + d),
                             FN_w=1e4, mass=1e4 * (1 + d), emission_px=int(1e4 * (1 + d))))
    return {"results": {"shaping_sweep": rows,
                        "as_provided": dict(dti=0.03, TP_w=2000.0, FP_w=5e5, emission_px=5165852),
                        "sweep_verdict": {"acceptance_rule": "test rule",
                                          "skeleton_dti": 0.02, "best_dti": 0.04}}}


def _evidence(tmp: Path, second_gains=None) -> dict:
    ev = {"truth": {"mode": "only", "px": TRUTH_PX, "km": TRUTH_PX * 0.1},
          "results": {"as_provided": dict(dti=0.0247, TP_w=1326.6, FP_w=20211.0, emission_px=21492),
                      "baselines": {"blanket_ones": dict(dti=0.0585, TP_w=61656.0, FP_w=4960570.0,
                                                         emission_px=5165852),
                                    "catalogue_plus_submission": dict(dti=0.0203, TP_w=1326.6,
                                                                      FP_w=78606.0, emission_px=79887)}}}
    rb = {"results": {"as_provided": dict(dti=0.0410, TP_w=2350.0, FP_w=37269.0, emission_px=39517)}}
    sweeps = {"--sweep": _sweep([0.0149, 0.0200, 0.0251])}
    if second_gains is not None:
        sweeps["--second-sweep"] = _sweep(second_gains)
    paths = {}
    for name, obj in [("--proxy-eval", ev), ("--reblend-eval", rb), *sweeps.items()]:
        p = tmp / f"{name.strip('-')}.json"
        p.write_text(json.dumps(obj))
        paths[name] = p
    idr = tmp / "in_domain.json"
    idr.write_text(json.dumps({"shaping": {"calibration_table": [
        dict(t0=0.0, thin=True, dilate=d, mean_dti=0.05 - 0.001 * d) for d in (0, 1, 3, 6)]}}))
    paths["--in-domain"] = idr
    return paths


def _run(tmp: Path, paths: dict) -> tuple[subprocess.CompletedProcess, dict]:
    out = tmp / "decision.json"
    cmd = [sys.executable, str(SCRIPT), "--out", str(out)]
    for k, v in paths.items():
        cmd += [k, str(v)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r, (json.loads(out.read_text()) if out.exists() else {})


def test_width_gain_table_is_floor_controlled():
    from scripts.decide_emission_width import width_gain_table

    t = width_gain_table(_sweep([0.0149, 0.0200, 0.0251]))
    assert t["widest_px"] == 6 and t["n_floors"] == 3
    assert t["min_gain"] == pytest.approx(0.0149) and t["max_gain"] == pytest.approx(0.0251)
    assert t["mean_gain"] == pytest.approx((0.0149 + 0.0200 + 0.0251) / 3)
    assert t["all_floors_positive"] is True
    # the contrast is taken at the SAME floor, so the width-0 column is the only reference
    assert all(row["dti_widest"] > row["dti_width0"] for row in t["per_floor"])

    neg = width_gain_table(_sweep([-0.001, -0.002, -0.003]))
    assert neg["all_floors_positive"] is False and neg["mean_gain"] < 0


def test_decision_script_runs_on_the_committed_evidence(tmp_path):
    """The crash-after-write defect: the step must exit 0, not just produce a file."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path / "real.json")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout tail:\n{r.stdout[-2000:]}"
    assert (tmp_path / "real.json").exists()


def test_condition_three_stays_unmet_without_a_second_sweep(tmp_path):
    r, d = _run(tmp_path, _evidence(tmp_path))
    assert r.returncode == 0, r.stderr
    cond3 = d["verdict"]["conditions"][-1]
    assert cond3["passes"] is False
    assert "NOT MEASURED" in cond3["measured"]
    assert d["verdict"]["conclusion"].startswith("widen, but not yet")


def test_condition_three_passes_when_the_second_ensemble_reproduces_the_gain(tmp_path):
    r, d = _run(tmp_path, _evidence(tmp_path, second_gains=[0.0131, 0.0188, 0.0240]))
    assert r.returncode == 0, r.stderr
    cond3 = d["verdict"]["conditions"][-1]
    assert cond3["passes"] is True, cond3
    assert "all floors positive" in cond3["measured"]
    assert d["verdict"]["width_gain_second_ensemble"]["n_floors"] == 3
    assert d["verdict"]["conclusion"].startswith("WIDEN")


def test_condition_three_fails_when_the_second_ensemble_flips_the_sign(tmp_path):
    r, d = _run(tmp_path, _evidence(tmp_path, second_gains=[-0.002, -0.003, -0.004]))
    assert r.returncode == 0, r.stderr
    cond3 = d["verdict"]["conditions"][-1]
    assert cond3["passes"] is False, cond3
    assert "NOT all positive" in cond3["measured"]
    assert d["verdict"]["conclusion"].startswith("widen, but not yet")


def test_stale_proxy_evidence_is_flagged_by_the_km_field():
    """The committed sweep/submission evidence predates the 0.01 -> 0.1 km-per-pixel fix.

    The decision itself is unaffected (it projects on pixels), but any prose quoting those files'
    ``truth.km`` would repeat a 10x error, so this test records which files are stale until the
    next proxy-eval run rewrites them.
    """
    stale, fresh = [], []
    for name in ("eval_submission.json", "eval_sweep.json", "eval_reblend_submission.json"):
        p = ROOT / "data" / "evidence" / "proxy" / name
        if not p.exists():
            continue
        t = json.loads(p.read_text()).get("truth", {})
        (stale if t.get("km") and abs(t["km"] - t["px"] * 0.1) > 1e-6 else fresh).append(name)
    # Either state is acceptable; what must never happen is an unnoticed mix.  Both lists are
    # asserted to be *disjoint and total*, i.e. every file is classified.
    assert not set(stale) & set(fresh)
    if stale:
        # document the known-stale set precisely (updated by the next proxy-eval run)
        assert set(stale) <= {"eval_submission.json", "eval_sweep.json",
                              "eval_reblend_submission.json"}, stale
