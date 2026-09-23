"""The generator section on docs/how_to_submit.html: it may only promise what it can measure.

The page now offers to build the submission .tif in the reader's browser. Two failure modes matter
more than the happy path:

* the payload goes STALE - docs/submission_meta.json pins an artifact hash; if the artifact in the
  tree no longer matches it, the page must render STALE (and the generator must be refused by its own
  self-checks), not keep printing the old provenance as though it were current;
* the section claims a verification that did not happen - if data/evidence/site_generator.json is
  absent or FAIL, the page must say so rather than borrow a previous run's green.

These are page-rendering tests (no node, no browser); the end-to-end behaviour of the same code is
tests/test_site_generator.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)


def _mod():
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    spec = importlib.util.spec_from_file_location("build_site_gen", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SHIPPED = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"


def _ev():
    def load(rel):
        q = ROOT / rel
        return json.loads(q.read_text()) if q.exists() else None
    return {"rules_quotes": load("data/evidence/rules_quotes.json"),
            "independent_verification": load("data/evidence/independent_verification.json"),
            "readiness": load("data/evidence/submission_readiness.json"),
            "site_generator": load("data/evidence/site_generator.json"),
            "site_payload": load("docs/submission_meta.json")}


def _text(mod, ev):
    import html as _html
    return _html.unescape(re.sub(r"<[^>]+>", " ", mod.build_how_to_submit(ev)))


# --------------------------------------------------------------------- structure
def test_the_generator_has_its_own_numbered_section_and_mount_point():
    mod = _mod()
    h = mod.build_how_to_submit(_ev())
    assert '<h2 id="generate">3.' in h, "the generator lost its section (and its anchor)"
    assert 'id="tif-generator"' in h, "the mount point the glue looks for is gone"
    # The JS is INLINED, not referenced: this repository's Pages build does not serve a newly added
    # docs/*.js (measured 2026-09-23 - the .json and .bin next to it return 200, the .js returns
    # GitHub's 404), so <script src> would ship a generator that only works from a clone.
    assert 'data-inlined-from="geotiff_writer.js"' in h
    assert 'data-inlined-from="generate_submission.js"' in h
    assert '<script src=' not in h, "a referenced script tag would 404 on the published site"
    # the writer must come before the glue that calls it
    assert h.index('data-inlined-from="geotiff_writer.js"') < h.index('data-inlined-from="generate_submission.js"')


def _inlined(h: str, name: str):
    m = re.search(r'<script data-inlined-from="' + re.escape(name) +
                  r'" data-sha256-12="([0-9a-f]{12})">\n(.*?)\n</script>', h, re.S)
    return (m.group(1), m.group(2).replace("<\\/script", "</script")) if m else (None, None)


def test_the_page_ships_exactly_the_code_the_tests_run():
    """Inlining is only honest if the inlined bytes ARE the file's bytes.

    The Node CLI and the DOM harness execute `docs/geotiff_writer.js` and `docs/generate_submission.js`
    from disk; a browser executes what the page carries. If those two ever differ, every green tick in
    tests/test_site_generator.py describes a file nobody downloads. So the page pins each block's
    sha256 (12 hex) AND this test compares the decoded block to the file, character for character.
    """
    import hashlib
    mod = _mod()
    h = mod.build_how_to_submit(_ev())
    for name in ("geotiff_writer.js", "generate_submission.js"):
        want = (ROOT / "docs" / name).read_text()
        digest, got = _inlined(h, name)
        assert got is not None, f"{name} is not inlined in the page"
        assert got == want, f"the page's inlined {name} differs from the file the tests run"
        assert digest == hashlib.sha256(want.encode()).hexdigest()[:12], \
            f"the page's own sha pin for {name} is wrong"


def test_a_page_with_no_js_on_disk_has_no_dead_script_reference():
    """build_site must not emit `<script src=...>` for a file it could not inline."""
    import shutil
    import subprocess
    tmp = ROOT / ".tmp-nofiles"
    (tmp / "docs").mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(ROOT / "scripts" / "build_site.py", tmp / "scripts_build_site.py")
        # the builder reads DOCS relative to its own ROOT; run it with an emptied docs dir instead
        # by pointing it at a scratch tree through the module, not by mutating this checkout.
        spec = importlib.util.spec_from_file_location("bs_scratch", ROOT / "scripts" / "build_site.py")
        bs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bs)
        saved = bs.DOCS
        bs.DOCS = tmp / "docs"
        try:
            assert bs.generator_tail() == "", "a missing writer still produced a script block"
        finally:
            bs.DOCS = saved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_routes_now_include_the_browser_and_the_workflow_and_stay_ordered():
    mod = _mod()
    h = mod.build_how_to_submit(_ev())
    for key in ("A", "B", "C", "D", "E", "F"):
        assert f"<b>{key}</b>" in h, f"route {key} is missing from the route table"
    assert "Six routes to that file" in h
    assert "make-submission.yml" in h, "route F must name the workflow it tells you to fire"
    assert "in your browser" in h.lower()


def test_route_commands_reference_files_that_exist():
    mod = _mod()
    cited = ["scripts/build_submission_payload.py", "docs/geotiff_writer.js",
             "docs/generate_submission.js", "docs/submission_meta.json",
             "docs/submission_field.bin", "scripts/check_site_generator.py",
             "scripts/package_submission.py", "data/evidence/site_generator.json",
             ".github/workflows/make-submission.yml"]
    h = mod.build_how_to_submit(_ev())
    for rel in cited:
        assert rel in h, f"the page no longer cites {rel}"
        assert (ROOT / rel).exists(), f"{rel} is cited by the page but missing from the repo"


# --------------------------------------------------------------------- provenance honesty
def test_provenance_numbers_come_from_the_payload_not_from_prose():
    mod, ev = _mod(), _ev()
    payload = ev["site_payload"]
    if not payload:
        pytest.skip("the site payload has not been built here")
    h = _text(mod, ev)
    assert f"{payload['encoding']['n_runs']:,}" in h, "the run count is not rendered from the manifest"
    assert f"{payload['blob']['bytes']:,}" in h, "the payload size is not rendered from the manifest"
    assert f"{payload['field']['one_px']:,}" in h, "the positive-pixel count is not rendered"
    assert str(payload["grid"]["epsg"]) in h
    assert payload["artifact"]["sha256"][:16] in h


def test_a_stale_payload_is_rendered_as_stale(monkeypatch):
    """The guard, not the happy path: flip the pin and the page must say STALE."""
    mod = _mod()
    ev = _ev()
    if not ev["site_payload"]:
        pytest.skip("the site payload has not been built here")
    ev["site_payload"] = dict(ev["site_payload"],
                              artifact=dict(ev["site_payload"]["artifact"],
                                            sha256="0" * 64))
    h = _text(mod, ev)
    assert "STALE" in h, "a payload that no longer matches the artifact must not look current"
    assert "IN SYNC" not in h


def test_the_generator_verification_table_reflects_its_evidence():
    mod, ev = _mod(), _ev()
    gen = ev["site_generator"]
    h = _text(mod, ev)
    if not gen:
        assert "not been executed in this checkout" in h, "an absent measurement must be visible"
        return
    assert f"{sum(1 for s in gen['steps'] if s['status'] == 'PASS')} measured steps passed" in h
    for s in gen["steps"]:
        assert s["name"] in h, f"step {s['name']} is missing from the rendered table"
    if gen["verdict"] == "PASS":
        assert "float32_bits_identical = true" in h
    else:
        assert gen["verdict"] in h


def test_a_fail_verdict_on_the_generator_is_visible_as_a_failure():
    mod, ev = _mod(), _ev()
    if not ev["site_generator"]:
        pytest.skip("no generator evidence to falsify")
    ev["site_generator"] = dict(ev["site_generator"], verdict="FAIL",
                                steps=ev["site_generator"]["steps"] +
                                [dict(name="injected", status="FAIL",
                                      measured=dict(reason="simulated"), source="test",
                                      note="simulated failure")])
    h = _text(mod, ev)
    assert "FAIL" in h and "injected" in h


# --------------------------------------------------------------------- claims table
def test_claims_table_names_the_generator_and_its_evidence_files():
    mod = _mod()
    h = _text(mod, _ev())
    tail = h[h.index("Every claim on this page"):]
    for frag in ("docs/submission_meta.json", "geotiff_writer.js", "site_generator.json"):
        assert frag in tail, f"the claims table lost {frag}"


def test_the_page_never_implies_the_file_was_uploaded():
    h = _text(_mod(), _ev())
    assert "HUMAN" in h, "the human-only gate must still be labelled as one"
    for forbidden in ("submission uploaded", "scored at", "we placed", "leaderboard rank of our"):
        assert forbidden not in h.lower(), f"the page claims an outcome ({forbidden}) it cannot know"


# --------------------------------------------------------------------- what the glue may not do
def test_the_glue_contains_no_typed_figures():
    """generate_submission.js must read every number from the manifest, so a rebuilt payload cannot
    leave a stale constant in the shipped JS."""
    js = (ROOT / "docs" / "generate_submission.js").read_text()
    # 3292 / 3730 / 32611 / 172974 / 532174 are the values a careless author would hardcode
    for lit in ("3292", "3730", "32611", "172974", "532174", "569531", "a3dcd6d5"):
        assert lit not in js, f"the glue hardcodes {lit} instead of reading submission_meta.json"


def test_the_writer_is_self_contained_and_free_of_network_calls():
    js = (ROOT / "docs" / "geotiff_writer.js").read_text()
    for forbidden in ("fetch(", "XMLHttpRequest", "import ", "require('http", "cdn."):
        assert forbidden not in js, f"the writer must not reach the network ({forbidden})"
    assert "require.main === module" in js, "the CLI entry point the tests use must exist"
