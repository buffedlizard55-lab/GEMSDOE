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
    # The renderer's contract, stated state-independently: WARN exactly when the evidence says the
    # check is incomplete.  (Asserting "never warn" would make this test fail whenever a quote is
    # genuinely wrong - it would police the evidence, not the renderer, and the evidence is policed
    # by the Verify-workflow job that produces it.)
    if s["all_found"]:
        assert "INCOMPLETE" not in html, "false INCOMPLETE warning on a fully verified report"
    else:
        assert f"INCOMPLETE ({s['n_found']}/{s['n_quotes']})" in html, \
            "missing quotes must be visible on the page"
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


def _proxy_ev():
    import json as _json
    root = ROOT / "data/evidence/proxy"
    def _l(name):
        p = root / name
        return _json.loads(p.read_text()) if p.exists() else None
    return {"proxy_stats": _l("proxy_stats.json"),
            "proxy_fetch": _l("fetch_meta.json"),
            "proxy_eval": _l("eval_reblend_submission.json") or _l("eval_submission.json"),
            "proxy_eval_combined": _l("eval_submission_combined.json"),
            "proxy_sweep": _l("eval_sweep.json")}


def test_proxy_section_renders_the_evidence_not_a_claim():
    """The proxy section is the site's only number taken on a population like the scored one.

    It must quote the committed evidence (feature count, split of the rasterised catalogue, the
    submission's DTI and the catalogue-copy baseline), link the publisher, and state the caveat -
    a page that showed the number without the caveat would be exactly the overclaiming this
    repository is written against.
    """
    ev = _proxy_ev()
    if not ev["proxy_stats"]:
        import pytest
        pytest.skip("proxy evidence not built in this checkout")
    mod = _site_mod()
    html = mod._proxy_catalogue(ev)
    ps = ev["proxy_stats"]
    assert f"{ps['proxy']['proxy_only_px']:,}" in html, "proxy-only pixel count missing"
    assert f"{ps['proxy']['mask_px']:,}" in html, "rasterised count missing"
    assert f"{(ev['proxy_fetch'] or {}).get('result', {}).get('features', 0):,}" in html, \
        "fetched-feature count missing"
    assert "doi.org/10.3133/ds1052" in html, "the SGMC publication link is not rendered"
    assert "catalogue" in html and "absent from the training labels" in html
    if ev["proxy_eval"]:
        d = ev["proxy_eval"]["results"]["as_submitted"]["dti"]
        assert f"{d:.4f}" in html, "the submission's proxy DTI is not rendered"
        cc = ev["proxy_eval"]["results"]["baselines"]["catalogue_copy"]["dti"]
        assert f"{cc:.4f}" in html, "the catalogue-copy baseline is not rendered"
    assert "not</em> the competition metric" in html, "the interpretation caveat is missing"
    assert len(html) > 2000


def test_proxy_section_without_evidence_says_so():
    mod = _site_mod()
    html = mod._proxy_catalogue({})
    assert "not available" in html.lower()
    assert "0.0247" not in html and "0.0" not in html.split("</h2>")[0], \
        "a missing proxy must not render numbers"
