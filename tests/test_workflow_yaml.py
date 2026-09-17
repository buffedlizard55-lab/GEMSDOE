"""Every GitHub workflow file must parse as YAML.

Why: the first firing of .github/workflows/place-competition-data.yml failed in
0 s with zero jobs because a step name contained an unquoted ``: `` ("Checkout
(full history: this job pushes a commit)"). GitHub rejected the file before
creating any job, and nothing in CI caught it. This test parses every workflow
so that class of error fails locally and in the Tests workflow instead of only
when a trigger is pushed.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))


def test_workflow_files_exist():
    assert len(WORKFLOWS) >= 8, "expected the established workflow set to be present"


def test_every_workflow_parses():
    bad = []
    for wf in WORKFLOWS:
        try:
            d = yaml.safe_load(wf.read_text())
        except yaml.YAMLError as e:
            bad.append(f"{wf.name}: {e}")
            continue
        if not isinstance(d, dict) or "jobs" not in d:
            bad.append(f"{wf.name}: no top-level 'jobs' mapping")
    assert not bad, "unparseable workflow files:\n" + "\n".join(bad)


def test_step_names_are_quoted_if_they_contain_colons():
    """A ``name:`` scalar containing ': ' must be quoted or colon-free.

    Catches the exact defect of run 35168727415 at the source level.
    """
    offenders = []
    for wf in WORKFLOWS:
        d = yaml.safe_load(wf.read_text())
        for job in d.get("jobs", {}).values():
            for step in job.get("steps", []) or []:
                name = step.get("name")
                if isinstance(name, str) and ": " in name:
                    # reload raw text to see whether it was quoted
                    raw = wf.read_text()
                    offenders.append(f"{wf.name}: {name!r}")
    assert not offenders, "step names with unquoted ': ':\n" + "\n".join(offenders)
