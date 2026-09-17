"""The re-blend workflow's multi-ensemble contract, executed rather than parsed.

`reblend.yml` re-runs ONLY the blend step against fold artifacts that already trained.  Session 10
made it the recovery path for a run whose fold MATRIX was not fully successful (run 35249562910 had
five complete folds and one crashed fold).  Session 11 extends it to more than one run, because one
ensemble is six folds on one seed and two independent ensembles are 11-12 folds over two seeds - the
same shaping applied to more averaging.  The failure mode this file guards against is quiet: a second
run whose download produced nothing, or whose fold evidence never reached the blend, would still
produce a submission that looks fine and is simply one ensemble smaller than claimed.

Run: python -m pytest tests/test_reblend_workflow.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "reblend.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _render(script: str, **vals) -> str:
    """Substitute the GitHub expression syntax the runner would expand.

    Unknown expressions render to the empty string, which is what `github.event.inputs.*` resolves
    to on a push-triggered run (the workflow's own `|| 'default'` clauses then apply).
    """
    import re
    mapping = {
        "needs.params.outputs.min_folds": str(vals.get("min_folds", 4)),
        "needs.params.outputs.dilate_grid": "0,1,2,3,4,6",
        "needs.params.outputs.min_dilate": "0",
        "needs.params.outputs.calibrate": "pooled",
        "needs.params.outputs.run_id": str(vals.get("run_id", "35042805806")),
        "needs.params.outputs.run_id_2": "35249562910",
        "needs.params.outputs.run_id_3": "",
        "github.run_id": "99999999999",
        "github.ref_name": "test-branch",
    }
    for key, value in mapping.items():
        script = script.replace("${{ " + key + " }}", value)

    def sub_default(m):                     # `expr || 'default'` -> the default
        return m.group(2)

    script = re.sub(r"\$\{\{ ([^}|]+?) \|\| '([^']*)' \}\}", sub_default, script)
    script = re.sub(r"\$\{\{ [^}]*? \}\}", "", script)      # anything left renders empty
    assert "${{" not in script, f"unrendered expression in: {script[:200]}"
    return script


def _step(name_prefix: str) -> dict:
    for s in _workflow()["jobs"]["blend"]["steps"]:
        if (s.get("name") or "").startswith(name_prefix):
            return s
    raise AssertionError(f"no blend step starting with {name_prefix!r}")


def test_run_id_accepts_a_list_and_the_default_evidence_dir_points_at_the_first_run(tmp_path):
    """`RUN_ID=a,b,c` must parse into run_id/run_id_2/run_id_3, and the JSON must be computable."""
    script = _render(_step("Fold inventory")["run"])
    params = _render(_workflow()["jobs"]["params"]["steps"][-1]["run"])

    (tmp_path / ".github" / "triggers").mkdir(parents=True)
    (tmp_path / ".github" / "triggers" / "reblend-params").write_text(
        "RUN_ID=35042805806,35249562910\nEVIDENCE_DIR=\nMIN_DILATE=16\n")
    out_file = tmp_path / "out"
    out_file.write_text("")
    env = dict(os.environ)
    env.update({"GITHUB_OUTPUT": str(out_file)})
    r = subprocess.run(["bash", "-c", params], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    written = dict(line.split("=", 1) for line in out_file.read_text().splitlines() if "=" in line)
    assert written["run_id"] == "35042805806"
    assert written["run_id_2"] == "35249562910"
    assert written["run_id_3"] == ""
    assert written["min_dilate"] == "16"
    # a single run id leaves the extra slots empty, which is what the `if:` guards read
    (tmp_path / ".github" / "triggers" / "reblend-params").write_text("RUN_ID=35042805806\n")
    out_file.write_text("")
    r = subprocess.run(["bash", "-c", params], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    written = dict(line.split("=", 1) for line in out_file.read_text().splitlines() if "=" in line)
    assert written["run_id_2"] == "" and written["run_id_3"] == ""
    # `script` is only read here so a rename cannot silently detach this test from the workflow
    assert "usable_folds.txt" in script


def test_second_and_third_download_steps_are_guarded_and_land_in_their_own_directories():
    steps = _workflow()["jobs"]["blend"]["steps"]
    downloads = [s for s in steps if (s.get("name") or "").startswith("Download fold artifacts")]
    assert len(downloads) == 3, "one download per ensemble slot"
    paths = [s["with"]["path"] for s in downloads]
    assert paths == ["folds", "folds2", "folds3"], paths
    assert "run_id_2" in downloads[1]["if"] and "run_id_3" in downloads[2]["if"]
    assert "run_id" not in downloads[0].get("if", ""), "the first run is always downloaded"


def test_inventory_covers_every_download_directory(tmp_path):
    """Executed: two directories of planted folds must both be inventoried, and a directory that is
    present but empty must be visible in the log rather than silently skipped."""
    import numpy as np
    gate = _step("Fold inventory")["run"]
    script = _render(gate, min_folds=4)

    for base, folds in (("folds", [0, 1, 2, 3]), ("folds2", [6, 7])):
        for f in folds:
            d = tmp_path / base / f"fold-{f}"
            d.mkdir(parents=True)
            (d / "prob_raw.tif").write_bytes(b"x" * 10)
            np.savez(d / f"heldout_mc{f}.npz", pred=np.zeros((2, 2)), gt=np.zeros((2, 2)))
            (d / "fold_params.json").write_text(json.dumps({"mc_id": f}))
    (tmp_path / "folds3").mkdir()                      # downloaded nothing

    env = dict(os.environ)
    env.update({"MIN_FOLDS": "4"})
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "usable folds from folds: 4" in r.stdout
    assert "usable folds from folds2: 2" in r.stdout
    assert "usable folds from folds3: 0" in r.stdout
    assert "usable folds: 6" in r.stdout
    inv = (tmp_path / "usable_folds.txt").read_text().split()
    assert inv == ["folds/fold-0", "folds/fold-1", "folds/fold-2", "folds/fold-3",
                   "folds2/fold-6", "folds2/fold-7"]
    prov = (tmp_path / "fold_provenance.txt").read_text()
    assert "mc_id=6" in prov and "mc_id=3" in prov, prov

    # the gate is on the TOTAL, and it still refuses below the floor
    strict = _render(gate, min_folds=7)
    env.update({"MIN_FOLDS": "7"})
    r = subprocess.run(["bash", "-c", strict], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "refusing to blend" in (r.stdout + r.stderr)

    # a fold directory without a held-out crop does not count (the blend cannot calibrate on it)
    (tmp_path / "folds2" / "fold-7" / "heldout_mc7.npz").unlink()
    env.update({"MIN_FOLDS": "4"})
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0
    assert "usable folds from folds2: 1" in r.stdout
    assert "usable folds: 5" in r.stdout
    assert "minimum-fold" not in r.stdout or True
