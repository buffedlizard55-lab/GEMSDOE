"""Cross-catalogue transfer (scripts/measure_cross_catalogue_transfer.py).

The measurement this repository was missing: emit catalogue A (USGS SGMC) and score it against an
INDEPENDENT catalogue B (USGS QFaults) code-2 pixels - B traces the training labels do not
contain.  The proxy population cannot do this because the proxy IS catalogue A: a catalogue copy
scores 0.0 there by construction.

Pinned here:
* the controls run BEFORE any transfer claim (labels-copy ~ 0, B-copy ~ 1, blanket-ones floor);
* the verdict is DERIVED - "ADOPT" is unreachable unless the controls pass, the union beats the
  model by more than the pre-registered margin, and the paired block bootstrap agrees;
* a union that genuinely helps is detected (a synthetic case built so the model misses B and A
  covers it), and a union that does not help is reported as "DO NOT ADOPT";
* missing or misaligned inputs fail loudly with the commands that build them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mod():
    spec = importlib.util.spec_from_file_location(
        "xcat", ROOT / "scripts" / "measure_cross_catalogue_transfer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TRANSFORM = from_origin(243350.0, 4508550.0, 100.0, 100.0)
CRS = "EPSG:32611"
H, W = 128, 160


def _write(path: Path, arr: np.ndarray, dtype: str) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=CRS, transform=TRANSFORM) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def _footprint() -> np.ndarray:
    fp = np.ones((H, W), bool)
    fp[:6, :] = False
    fp[-6:, :] = False
    return fp


def _base(tmp_path: Path, a_covers_b: bool) -> dict:
    """Synthetic world: labels far from B, A either covering B's new traces or somewhere else."""
    fp = _footprint()

    labels = np.zeros((H, W), np.uint8)
    labels[20:22, 20:140] = 1                      # catalogue fault, far from everything below

    b = np.zeros((H, W), np.uint8)
    b[60:62, 30:150] = 2                           # B trace with no label within R -> code 2
    b[100:110, 40:42] = 2                          # a second, shorter B-only trace
    b[20:22, 20:60] = 1                            # B trace the labels already have -> code 1

    a = np.zeros((H, W), np.uint8)
    if a_covers_b:
        a[61:63, 30:150] = 2                       # A runs alongside the first B trace (within R)
        a[101:103, 40:60] = 2                      # and partly alongside the second
    else:
        a[80:82, 10:90] = 2                        # A is somewhere else entirely

    pred = np.full((H, W), np.nan, np.float32)
    body = np.zeros((H, W), np.float32)
    body[19:23, 19:141] = 1.0                      # the model restates the catalogue
    pred[fp] = body[fp]

    return dict(
        pred=_write(tmp_path / "pred.tif", pred, "float32"),
        labels=_write(tmp_path / "labels.tif", np.where(fp, labels, 255).astype(np.uint8), "uint8"),
        a=_write(tmp_path / "a.tif", a, "uint8"),
        b=_write(tmp_path / "b.tif", b, "uint8"),
        template=_write(tmp_path / "template.tif", np.where(fp, 0.0, np.nan).astype(np.float32),
                        "float32"),
    )


def _argv(files, out, **kw):
    return ["--truth-b", str(files["b"]), "--catalogue-a", str(files["a"]),
            "--pred", str(files.get("pred", "")) if files.get("pred") else "",
            "--labels", str(files["labels"]), "--template", str(files["template"]),
            "--widths", kw.pop("widths", "0,2"), "--a-codes", kw.pop("a_codes", "2"),
            "--block-px", str(kw.pop("block_px", 32)),
            "--bootstraps", str(kw.pop("bootstraps", 200)),
            "--min-union-gain", str(kw.pop("min_gain", 0.01)),
            "--min-bootstrap-prob", str(kw.pop("min_prob", 0.95)),
            "--out", str(out), "--quiet", *[x for kv in kw.items() for x in (f"--{kv[0]}", str(kv[1]))]]


# --------------------------------------------------------------------------------------
# controls and structure
# --------------------------------------------------------------------------------------
def test_report_structure_controls_and_populations(tmp_path):
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    out = tmp_path / "report.json"
    assert m.main(_argv(files, out)) == 0
    rep = json.loads(out.read_text())
    assert rep["generated_by"] == "scripts/measure_cross_catalogue_transfer.py"
    assert rep["controls_pass"] is True
    assert rep["controls"]["labels_copy_vs_B_only"]["global_dti"] < 0.05, \
        "a copy of the training labels must earn ~0 against B-only, or B-only is not 'new'"
    assert rep["controls"]["B_copy_vs_B_only"]["global_dti"] > 0.99, \
        "a copy of B must score 1.0 against B-only, or the scorer is broken on this population"
    assert 0.0 < rep["controls"]["blanket_ones_vs_B_only"]["global_dti"] < 0.2
    # populations
    assert rep["overlap"]["B_only_px"] == 2 * 120 + 10 * 2
    assert rep["overlap"]["B_near_label_px"] == 80
    assert rep["overlap"]["B_near_label_fraction_of_all"] == pytest.approx(
        80 / (80 + 2 * 120 + 10 * 2), rel=1e-6)
    assert rep["overlap"]["B_only_within_R_of_A_px"] > 0
    assert rep["overlap"]["B_only_within_R_of_A_px"] <= rep["overlap"]["B_only_px"]
    assert 0.0 <= rep["overlap"]["B_only_recall_by_A_at_R"] <= 1.0
    assert rep["overlap"]["components_B_only"]["components"] == 2
    # measurement kinds
    kinds = [r["kind"] for r in rep["measurements"]]
    assert kinds.count("model") == 1
    assert kinds.count("catalogue_a") == 2 and kinds.count("union") == 2
    assert set(rep["bootstrap"]["union_vs_model"]) >= {"model_as_shipped"}


def test_verdict_adopts_a_union_that_really_helps(tmp_path):
    """Model restates the catalogue and misses B; A covers B -> the union must be adopted."""
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    out = tmp_path / "report.json"
    assert m.main(_argv(files, out, bootstraps=400)) == 0
    rep = json.loads(out.read_text())
    v = rep["verdict"]
    model_dti = v["model_dti"]
    assert model_dti < 0.05, "the synthetic model only restates the catalogue, so it must score ~0"
    assert v["union_gain"] > 0.01
    assert v["criterion"]["controls_pass"] is True
    assert v["criterion"]["bootstrap_prob_beats_model"] >= 0.95
    assert v["conclusion"].startswith("ADOPT"), v["conclusion"]


def test_verdict_refuses_a_union_that_does_not_help(tmp_path):
    """Same world, but A's traces are elsewhere: no gain, so no adoption."""
    m = _mod()
    files = _base(tmp_path, a_covers_b=False)
    out = tmp_path / "report.json"
    assert m.main(_argv(files, out, bootstraps=400)) == 0
    rep = json.loads(out.read_text())
    v = rep["verdict"]
    assert v["union_gain"] <= 0.01 or (v["criterion"]["bootstrap_prob_beats_model"] or 0) < 0.95
    assert v["conclusion"].startswith("DO NOT ADOPT"), v["conclusion"]
    assert rep["overlap"]["B_only_within_R_of_A_px"] == 0


def test_conclusion_is_never_adopt_when_a_control_fails(tmp_path):
    """If B-only is not independent of the labels, no transfer claim may be made."""
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    # make B's "new" traces sit ON the training labels: code 2 pixels within R of a label
    with rasterio.open(files["b"]) as src:
        b = src.read(1)
    labels = rasterio.open(files["labels"]).read(1)
    b[(labels > 0)] = 2                            # B-only now coincides with the catalogue
    _write(files["b"], b, "uint8")
    out = tmp_path / "report.json"
    assert m.main(_argv(files, out)) == 0
    rep = json.loads(out.read_text())
    assert rep["controls_pass"] is False, "labels-copy scored too well: the control must fire"
    assert rep["verdict"]["conclusion"].startswith("REFUSED"), rep["verdict"]["conclusion"]


def test_measurement_without_a_prediction_is_reported_as_not_measurable(tmp_path):
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    files_nopred = {k: v for k, v in files.items() if k != "pred"}
    out = tmp_path / "report.json"
    assert m.main(_argv(files_nopred, out)) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"]["conclusion"].startswith("NOT MEASURABLE")
    kinds = [r["kind"] for r in rep["measurements"]]
    assert "model" not in kinds and "union" not in kinds and kinds.count("catalogue_a") == 2


def test_widening_a_grows_the_emission_and_the_false_positive_mass(tmp_path):
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    out = tmp_path / "report.json"
    assert m.main(_argv(files, out, widths="0,3,6")) == 0
    rep = json.loads(out.read_text())
    rows = {r["label"]: r for r in rep["measurements"] if r["kind"] == "catalogue_a"}
    px = [rows[k]["emitted_px"] for k in sorted(rows, key=lambda s: rows[s]["width"])]
    fp = [rows[k]["FP_w"] for k in sorted(rows, key=lambda s: rows[s]["width"])]
    assert px[0] < px[1] < px[2], f"widening must grow the emission: {px}"
    assert fp[0] < fp[2], f"widening must grow the false-positive mass: {fp}"


def test_a_codes_must_include_the_independent_part(tmp_path):
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    with pytest.raises(SystemExit) as exc:
        m.main(_argv(files, tmp_path / "never.json", a_codes="1"))
    assert "code 2" in str(exc.value)
    assert not (tmp_path / "never.json").exists()


# --------------------------------------------------------------------------------------
# loud failures
# --------------------------------------------------------------------------------------
def test_missing_catalogue_b_prints_the_commands_that_build_it(tmp_path):
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    files["b"] = tmp_path / "absent.tif"
    with pytest.raises(SystemExit) as exc:
        m.main(_argv(files, tmp_path / "never.json"))
    msg = str(exc.value)
    assert "fetch_qfaults.py" in msg and "build_proxy_catalogue.py" in msg
    assert "cross-catalogue.yml" in msg


def test_grid_mismatch_is_refused(tmp_path):
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    small = _write(tmp_path / "small_b.tif", np.zeros((64, 80), np.uint8), "uint8")
    files["b"] = small
    with pytest.raises(SystemExit) as exc:
        m.main(_argv(files, tmp_path / "never.json"))
    assert "not on the prediction grid" in str(exc.value)


def test_empty_b_only_population_is_refused(tmp_path):
    m = _mod()
    files = _base(tmp_path, a_covers_b=True)
    with rasterio.open(files["b"]) as src:
        b = src.read(1)
    _write(files["b"], np.where(b == 2, 1, b).astype(np.uint8), "uint8")   # every trace is code 1
    with pytest.raises(SystemExit) as exc:
        m.main(_argv(files, tmp_path / "never.json"))
    assert "no code-2 pixels" in str(exc.value)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def test_band_growth_is_euclidean_and_idempotent_at_zero():
    m = _mod()
    mask = np.zeros((32, 32), bool)
    mask[16, 16] = True
    assert m.band(mask, 0).sum() == 1
    b1 = m.band(mask, 1)
    b3 = m.band(mask, 3)
    assert b1.sum() == 5, "radius 1 must be the 4-neighbourhood plus the seed"
    assert b3.sum() == int(np.count_nonzero(b3 > 0))
    # a Euclidean disk of radius 3 has 29 pixels
    assert b3.sum() == 29
    empty = m.band(np.zeros((8, 8), bool), 4)
    assert empty.sum() == 0


def test_union_field_is_a_probability_maximum_not_a_mask_or():
    m = _mod()
    a = np.array([[0.2, 0.0], [0.9, 0.4]])
    b = np.array([[0.5, 0.1], [0.0, 0.4]])
    np.testing.assert_allclose(m.union_field(a, b), [[0.5, 0.1], [0.9, 0.4]])
    nan = np.array([[np.nan, 0.3]])
    np.testing.assert_allclose(m.union_field(nan, np.array([[0.2, 0.0]])), [[0.2, 0.3]])
