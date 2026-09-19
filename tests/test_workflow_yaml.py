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


# --------------------------------------------------------------------------------------
# parameter-file plumbing
#
# Run 35412306524 died in 28 s with `argument --bootstraps: invalid int value: ''`.  The params job
# had read the override line
#
#     [ -n "$(v BOOTSTRAPS)" ] && BOOT="$(v BOOT)" || true
#
# which tests one key and assigns from a DIFFERENT one.  `BOOT` is not a key in
# .github/triggers/cross-catalogue-params, so the substitution came back empty and the empty string
# was handed to a typed argparse argument.  The defect was invisible for a whole session because the
# first run failed earlier, in the qfaults job, so the transfer job never started.
#
# Both tests below are static: they read the workflow text and the params file and need no runner.
# --------------------------------------------------------------------------------------
import re  # noqa: E402

_OVERRIDE = re.compile(r'\[\s*-n\s*"\$\(v\s+([A-Z][A-Z0-9_]*)\)\s*"\s*\]\s*&&\s*'
                       r'([A-Z][A-Z0-9_]*)="\$\(v\s+([A-Z][A-Z0-9_]*)\)"')
_PARAMS_FILE = re.compile(r'\.github/triggers/([a-z0-9-]+)-params')
_LOOKUP = re.compile(r'\$\(v\s+([A-Z][A-Z0-9_]*)\)')


def _params_keys(wf: Path) -> tuple[set[str], list[str]]:
    """Keys this workflow may read: set in its params file, or documented as optional in its header.

    Optional overrides (proxy-eval's SHAPING_GRID, reblend's DILATE_GRID) are deliberately absent
    from the params file - an absent key means "use the script default" - but they are always named
    in the workflow's own comment block, so a reader can discover them.  A key that is neither set
    nor documented is a typo or a rename, and yields an empty string.
    """
    text = wf.read_text()
    keys, files = set(), []
    for name in sorted(set(_PARAMS_FILE.findall(text))):
        pf = REPO / ".github" / "triggers" / f"{name}-params"
        files.append(pf.name)
        if pf.exists():
            keys |= set(re.findall(r"^([A-Z][A-Z0-9_]*)=", pf.read_text(), re.M))
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            keys |= set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", line))
    return keys, files


def test_parameter_overrides_read_the_key_they_test():
    """`[ -n "$(v KEY)" ] && VAR="$(v KEY)"` - the tested key and the read key must be the same.

    Reading a different key silently yields an empty string, which either overrides a good default
    with nothing (the run above) or passes an empty value to a typed CLI argument.
    """
    offenders = []
    for wf in WORKFLOWS:
        for i, line in enumerate(wf.read_text().splitlines(), 1):
            for tested, _var, read in _OVERRIDE.findall(line):
                if tested != read:
                    offenders.append(f"{wf.name}:{i}: tests {tested} but assigns from {read}")
    assert not offenders, "parameter overrides that read a key other than the one they test:\n" + "\n".join(offenders)


def test_every_parameter_key_read_in_an_assignment_exists_in_its_params_file():
    """A `v KEY` on the right-hand side of an assignment must be a key the params file actually has.

    This is the same defect seen from the other end: it also catches a renamed key in the params file
    (or a workflow copied from another one) where the guard and the read agree with each other but
    neither matches the file, which again produces an empty value.
    """
    offenders = []
    for wf in WORKFLOWS:
        keys, files = _params_keys(wf)
        if not files:
            continue
        for i, line in enumerate(wf.read_text().splitlines(), 1):
            if "=" not in line or not re.search(r'[A-Z][A-Z0-9_]*="\$\(v\s', line):
                continue
            for read in _LOOKUP.findall(line.split("=", 1)[1]):
                if read not in keys:
                    offenders.append(f"{wf.name}:{i}: reads {read}, absent from {'/'.join(files)}")
    assert not offenders, "assignments reading keys the params file does not define:\n" + "\n".join(offenders)


def test_typed_cli_arguments_are_never_fed_a_parameter_that_can_be_empty():
    """An output interpolated straight into a `--flag` must be able to be non-empty on a push run.

    Push-triggered runs carry no `github.event.inputs`, so an expression without a fallback yields
    '' - which is what killed run 35412306524 (`argument --bootstraps: invalid int value: ''`).
    An *explicit* `|| ''` is different: that is an author saying "unset", and every such output here
    is consumed through a shell variable or an `if:`/`[ -n ]` guard rather than passed to a flag, so
    only unguarded direct interpolations are linted.
    """
    offenders = []
    for wf in WORKFLOWS:
        text = wf.read_text()
        d = yaml.safe_load(text)
        jobs = d.get("jobs", {}) or {}
        params_job = next((j for j in jobs.values() if isinstance(j.get("outputs"), dict)
                           and j.get("outputs")), None)
        if params_job is None:
            continue
        declared = set(params_job["outputs"].keys())
        consumed = set(re.findall(r"needs\.params\.outputs\.([a-z0-9_]+)", text))
        undeclared = sorted(consumed - declared)
        if undeclared:
            offenders.append(f"{wf.name}: consumes params outputs never declared: {undeclared}")
        step_text = "\n".join(str(st.get("run", "")) for st in params_job.get("steps", []) or [])
        pkeys, _ = _params_keys(wf)
        for out in sorted(consumed & declared):
            m = re.search(rf'echo "{out}=\$([A-Z][A-Z0-9_]*)"', step_text)
            if not m:
                offenders.append(f"{wf.name}: output {out} is declared but never written by the params step")
                continue
            var = m.group(1)
            init = re.search(rf'\b{var}="([^"]*)"', step_text)   # several may share a line
            if not init:
                offenders.append(f"{wf.name}: {var} (output {out}) is never initialised")
                continue
            body = init.group(1)
            if not re.search(r"\$\{\{", body):
                continue                       # a plain literal - it cannot arrive empty
            fallback = re.search(r"\|\|\s*'([^']*)'", body)
            override = re.search(rf'\[\s*-n\s*"\$\(v\s+([A-Z][A-Z0-9_]*)\)\s*"\s*\]\s*&&\s*'
                                 rf'{var}="\$\(v\s+[A-Z][A-Z0-9_]*\)"', step_text)
            can_be_empty = not ((fallback and fallback.group(1)) or
                                (override and override.group(1) in pkeys))
            if not can_be_empty:
                continue
            direct = re.search(rf'--[a-z0-9-]+\s+"?\$\{{\{{\s*needs\.params\.outputs\.{out}\s*\}}\}}"?',
                               text)
            guarded = bool(re.search(rf'\[\s*-n\s+"?\$\{{\{{\s*needs\.params\.outputs\.{out}', text)) or \
                bool(re.search(rf'if:.*needs\.params\.outputs\.{out}\s*!=', text))
            if direct and not guarded:
                offenders.append(f"{wf.name}: {out} can be empty on a push-triggered run and is passed "
                                 f"straight to a command-line flag")
    assert not offenders, "parameters that can reach a typed CLI argument empty:\n" + "\n".join(offenders)
