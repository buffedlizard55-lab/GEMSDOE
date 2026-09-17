"""The ensemble workflow's fold-identity contract, executed (not just parsed).

Why this file exists
--------------------
Actions run 35170395055 (2026-09-17, the queued SECOND 6-fold ensemble) had all six folds train
for ~3 hours and then threw every one of them away:

* the `Train fold` step overrode only ``training.mc_id=${{ matrix.fold }}`` (0..5), ignoring
  ``FOLD_OFFSET``, so ``src/train.py`` wrote ``heldout_mc0..5.npz`` / ``model_mc0..5_*.pt``;
* the `Stage fold outputs` step copied ``outputs/heldout_mc$((FOLD_OFFSET + matrix.fold)).npz``
  -> with ``FOLD_OFFSET=6`` that path never existed, ``cp`` failed, the step failed on every
  fold, `Upload fold artifact` was skipped (no ``if: always()``), and the blend job correctly
  refused to blend an incomplete fold set;
* the failure record could not even be committed, because the `Checkout` step of the blend job
  had been skipped (it had no ``if: always()``) and the ``if: always()`` evidence step then ran
  without a ``.git`` directory.

The previous test only checked that workflows *parse* (``tests/test_workflow_yaml.py``).  This
file goes further: it extracts the real ``run:`` shell from the workflow YAML and executes it
against planted artefacts that follow the naming contract of ``src/train.py``.  The positive test
plants ``heldout_mc6.npz`` with ``MC_ID=6``; the negative control plants ``heldout_mc0.npz`` and
requires the step to fail loudly.  A silent mismatch is the failure mode that cost 18 CPU-hours.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "train-ensemble.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _step(job: str, name: str) -> dict:
    for s in _workflow()["jobs"][job]["steps"]:
        if s.get("name") == name:
            return s
    raise AssertionError(f"no step named {name!r} in job {job!r}")


def _render(script: str, matrix_fold: int = 0) -> str:
    """Substitute the GitHub expression syntax the runner would expand."""
    return (script.replace("${{ matrix.fold }}", str(matrix_fold))
                  .replace("${{ github.run_id }}", "99999999999")
                  .replace("${{ github.sha }}", "deadbeef")
                  .replace("${{ github.ref_name }}", "test-branch"))


def _plant_outputs(tmp: Path, mc_in_manifest: int, heldout_mc: int) -> None:
    """Write exactly what src/train.py + src/inference.py write for a fold."""
    out = tmp / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prob_raw.tif").write_bytes(b"\x00" * 64)          # existence is what staging checks
    (out / "manifest.json").write_text(json.dumps(
        {"models": [{"file": f"model_mc{mc_in_manifest}_unetplusplus_resnet34.pt",
                     "arch": "unetplusplus", "encoder": "resnet34", "mc": mc_in_manifest, "dti": 0.1}]}))
    np.savez_compressed(out / f"heldout_mc{heldout_mc}.npz",
                        pred=np.zeros((4, 4), np.float16), gt=np.zeros((4, 4), np.uint8))
    (out / f"model_mc{mc_in_manifest}_unetplusplus_resnet34.pt").write_bytes(b"\x00" * 128)
    (out / "train.log").write_text("train\n")
    (out / "inference.log").write_text("inference\n")


def _run_staging(tmp: Path, env_extra: dict, matrix_fold: int = 0) -> subprocess.CompletedProcess:
    import os

    script = _render(_step("fold", "Stage fold outputs")["run"], matrix_fold=matrix_fold)
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", "-c", script], cwd=tmp, env=env,
                          capture_output=True, text=True)


# ---------------------------------------------------------------- the executed contract
def test_staging_accepts_the_trainer_naming_with_an_offset(tmp_path):
    """FOLD_OFFSET=6 + matrix.fold=0 -> the trainer's mc id is 6; staging must find mc6."""
    _plant_outputs(tmp_path, mc_in_manifest=6, heldout_mc=6)
    r = _run_staging(tmp_path, {"MC_ID": "6", "FOLD_OFFSET": "6", "ENSEMBLE_SEED": "43",
                                "GITHUB_SHA": "deadbeef"})
    assert r.returncode == 0, f"staging failed:\n{r.stdout}\n{r.stderr}"
    fold = tmp_path / "fold"
    names = sorted(p.name for p in fold.iterdir())
    assert "prob_raw.tif" in names and "manifest.json" in names
    assert "heldout_mc6.npz" in names, names
    assert any(n.startswith("model_mc6_") for n in names), names
    params = json.loads((fold / "fold_params.json").read_text())
    assert params["mc_id"] == 6 and params["fold_offset"] == 6 and params["ensemble_seed"] == 43


def test_staging_refuses_a_fold_whose_identity_does_not_match(tmp_path):
    """The exact run-35170395055 defect: job says mc6, the trainer wrote mc0.

    Staging must fail loudly instead of producing a fold the blend job cannot calibrate on.
    """
    _plant_outputs(tmp_path, mc_in_manifest=0, heldout_mc=0)
    r = _run_staging(tmp_path, {"MC_ID": "6", "FOLD_OFFSET": "6", "ENSEMBLE_SEED": "43",
                                "GITHUB_SHA": "deadbeef"})
    assert r.returncode != 0, "a fold identity mismatch must fail the step"
    assert "mc ids [0] != workflow fold identity 6" in (r.stdout + r.stderr)


def test_staging_refuses_a_missing_heldout_crop(tmp_path):
    """No calibration crop -> the blend job's pooled shaping search is impossible; fail here."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True)
    (out / "prob_raw.tif").write_bytes(b"\x00" * 64)
    (out / "manifest.json").write_text(json.dumps({"models": [{"mc": 6}]}))
    r = _run_staging(tmp_path, {"MC_ID": "6", "FOLD_OFFSET": "6", "ENSEMBLE_SEED": "43",
                                "GITHUB_SHA": "deadbeef"})
    assert r.returncode != 0
    assert "no outputs/heldout_mc*.npz" in (r.stdout + r.stderr)


# ---------------------------------------------------------------- the parameters step
def test_parameters_step_computes_the_offset_fold_identity(tmp_path):
    (tmp_path / ".github" / "triggers").mkdir(parents=True)
    (tmp_path / ".github" / "triggers" / "ensemble-params").write_text(
        "# a comment\nFOLD_OFFSET=6\nENSEMBLE_SEED=43\n")
    env_file = tmp_path / "github_env"
    env_file.touch()
    import os
    script = _render(_step("fold", "Read ensemble parameters (committed trigger file overrides the defaults)")["run"],
                     matrix_fold=2)
    env = dict(os.environ)
    env["GITHUB_ENV"] = str(env_file)
    env["FOLD_OFFSET"] = "0"      # stale job-level default: must NOT win over the trigger file
    env["ENSEMBLE_SEED"] = "42"
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    written = dict(line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line)
    assert written["FOLD_OFFSET"] == "6", written
    assert written["ENSEMBLE_SEED"] == "43", written
    assert written["MC_ID"] == "8", written                       # 6 + matrix.fold(2)


def test_train_step_applies_the_workflow_fold_identity():
    run = _step("fold", "Train fold ${{ matrix.fold }}")["run"]
    assert 'training.mc_id="$MC_ID"' in run, run
    assert 'training.seed="$ENSEMBLE_SEED"' in run, run
    # the raw matrix index must not be used as the fold identity anywhere in the trainer call
    assert "training.mc_id=${{ matrix.fold }} " not in run


def test_no_step_hardcodes_a_heldout_file_name():
    """Staging must glob the trainer's output; a composed name is what broke run 35170395055."""
    run = _step("fold", "Stage fold outputs")["run"]
    assert "heldout_mc$(( FOLD_OFFSET" not in run
    assert "heldout_mc*.npz" in run


# ---------------------------------------------------------------- recoverability of a failure
def test_fold_artifact_upload_runs_even_when_staging_fails():
    """Three hours of training must survive a staging hiccup: `if: always()` on the upload."""
    up = _step("fold", "Upload fold artifact")
    assert up.get("if") == "always()", up.get("if")
    assert up["with"]["path"] == "fold/"


def test_blend_job_checkout_runs_before_the_refusal_check():
    """A failed ensemble must still be able to commit FAILED.json (needs a .git directory)."""
    steps = _workflow()["jobs"]["blend"]["steps"]
    assert steps[0].get("name") == "Checkout", [s.get("name") for s in steps]
    assert steps[0].get("if") == "always()", steps[0].get("if")
    assert steps[1].get("name") == "Refuse to blend incomplete fold set"


def test_blend_commit_step_pushes_with_retries_and_a_hard_failure():
    run = _step("blend", "Commit run report + submission to the branch")["run"]
    assert "fold_params.json" in run, "per-fold identity must be committed with the run record"
    assert run.count("git push") >= 1
    assert 'pushed="yes"' in run and "FATAL" in run, "an unpushable record must fail the step"
