"""Regression tests for the generated site (2026-09-16, session 6).

The site is generated FROM the evidence JSON; these tests pin the rendering contract so a
schema drift fails loudly instead of shipping a false statement.  The case that motivated
them: scripts/verify_rules_quotes.py reshaped its report from `verification`/`document` to
`summary`/`source`, and build_site._scoring_universe kept reading the old keys — so the
live metric page printed "Quotation verification is INCOMPLETE (0/0)" next to a table of 19
verified quotes, contradicting the overview page that reads the same file correctly.

Stdlib + numpy-free: build_site.py imports only datetime/html/json/pathlib.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _site_mod():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evidence_quotes():
    return json.loads((ROOT / "data/evidence/rules_quotes.json").read_text())


def test_scoring_universe_badge_matches_committed_evidence():
    """The badge must quote the committed report's own counts — never a false 0/0 — and must
    link to the publisher, never to a local extraction path."""
    mod = _site_mod()
    rq = _evidence_quotes()
    html = mod._scoring_universe({"rules_quotes": rq})
    s = rq["summary"]
    assert f"{s['n_found']}/{s['n_quotes']}" in html, "badge does not quote the evidence counts"
    assert "INCOMPLETE" not in html, "false INCOMPLETE warning on verified quotes"
    assert "/tmp/" not in html, "a local extraction path leaked into a published link"
    assert ("docs.nlr.gov" in html) or ("www.nlr.gov" in html), "no rules-PDF link rendered"
    for qid in ("phase1_target", "phase2_target", "labels_source", "ranking_basis"):
        assert qid in html, f"key quote {qid} missing from the rendered table"


def test_scoring_universe_warns_only_when_quotes_are_actually_missing():
    """The INCOMPLETE warning must survive for genuinely incomplete reports — the session-6
    fix must not have been to delete the warning, but to read the right keys."""
    mod = _site_mod()
    rq = _evidence_quotes()
    rq["quotes"][0]["exact_match"] = False
    rq["summary"] = dict(rq["summary"], n_found=rq["summary"]["n_found"] - 1, all_found=False)
    html = mod._scoring_universe({"rules_quotes": rq})
    n = rq["summary"]
    assert f"INCOMPLETE ({n['n_found']}/{n['n_quotes']})" in html


def test_scoring_universe_without_evidence_says_so():
    """No evidence file -> an explicit 'not verified' note, not a wolf-crying 0/0."""
    mod = _site_mod()
    html = mod._scoring_universe({})
    assert "not been machine-verified" in html
    assert "INCOMPLETE" not in html
