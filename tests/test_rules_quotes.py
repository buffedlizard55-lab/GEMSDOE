"""Tests for the rules-quote verifier (scripts/verify_rules_quotes.py).

Why: this script is the machine check behind the single most consequential claim in the repository -
that both prize phases score the expert-mapped NEW fault dataset while the training labels are the
*existing* catalogue (§1.1, §3.3).  Two things must hold for that claim to be trustworthy:

  1. the normalisation must not be so loose that a paraphrase would pass (no fuzzy matching), and
     must not be so strict that the same sentence fails merely because a PDF line break moved;
  2. a MISS must arrive with the document's own words next to the quoted ones, because the job log
     is not readable from the development sandbox (the Actions log host is blocked) - a silent "not
     found" is exactly the kind of unmeasurable claim this repository exists to avoid.

Offline: normalisation and diagnosis are pure string functions; the committed report is read from
disk rather than regenerated (a PDF fetch needs a runner - docs.nlr.gov is not on the sandbox
allowlist).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("vrq", ROOT / "scripts" / "verify_rules_quotes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalisation_folds_typography_but_not_language():
    m = _mod()
    doc = m.norm("The labels\u2019 owner said \u201cyes\u201d \u2014 then left\u00a0the  room.")
    assert doc == "The labels' owner said \"yes\" - then left the room."
    # a hyphen at a line break is a wrapped word, not a hyphen
    assert m.norm("over-\nlap here") == "overlap here"
    # ...but a real hyphen survives
    assert m.norm("100-m resolution") == "100-m resolution"
    # a changed word must NOT match after normalisation (no fuzzy matching anywhere)
    assert "the labels own" not in doc


def test_divergence_points_at_the_exact_disagreement():
    m = _mod()
    text = m.norm("Intro. The set of faults included in the public test dataset and the rest.")
    quote = m.norm("The set of faults included in the public test datasets and the rest.")
    d = m.divergence(quote, text)
    assert d["prefix_len"] == len("The set of faults included in the public test dataset")
    assert d["quote_continues"].startswith("s and the rest")
    assert d["doc_continues"].startswith(" and the rest")
    assert d["doc_offset"] > 0


def test_divergence_on_a_total_miss_is_empty_not_a_crash():
    m = _mod()
    d = m.divergence(m.norm("nothing like this is in the document"), m.norm("other text"))
    assert d["prefix_len"] == 0 and d["doc_continues"] is None


def test_committed_report_matches_the_quote_list():
    """The published evidence must be about the same sentences the script checks.

    If a quote is edited or added without re-running the workflow, the site would render a count
    that no longer corresponds to the list - so the schema agreement is pinned here.  (Whether each
    quote was FOUND is the runner's business; it needs the PDF.)
    """
    m = _mod()
    report = json.loads((ROOT / "data/evidence/rules_quotes.json").read_text())
    ids = [q[0] for q in m.QUOTES]
    assert len(ids) == len(set(ids)), "duplicate quote ids"
    published = {q["id"]: q["quote"] for q in report["quotes"]}
    missing = [i for i in ids if i not in published]
    assert not missing, f"quotes not present in the committed evidence: {missing}"
    mismatched = [i for i, _, q, _ in m.QUOTES if published[i] != q]
    assert not mismatched, f"quote text changed since the evidence was produced: {mismatched}"
    assert report["summary"]["n_quotes"] == len(ids)
    # A miss must carry its diagnosis - but only reports written by a diagnostic-capable run can,
    # and the committed report may predate that feature (the workflow refreshes it).  The marker is
    # `extracted_text`, which the script only writes when it also writes diagnostics.
    diagnostic_capable = report.get("extracted_text") is not None
    for q in report["quotes"]:
        if not q["exact_match"] and diagnostic_capable:
            assert q.get("diagnostic", {}).get("prefix_len") is not None, \
                f"{q['id']} failed without a diagnostic"
    assert report["source"]["canonical_url"] == m.RULES_URL, "evidence must link the publisher"


def test_page_furniture_is_removed_from_page_margins_only():
    """A bare page number at a page boundary is furniture, not prose.

    MEASURED 2026-09-16: the official rules' §3.6.2 sentence continues across a page break, and the
    extractor emitted the next page's number inside it ("... and the 12 relative weight of faults
    ..."), so a genuinely verbatim quotation could not match.  The strip must be exactly as narrow as
    that: bare numbers at the very start or end of a page, nothing inside the text.
    """
    m = _mod()
    assert m.strip_page_furniture("12\nrelative weight of faults\n13", 12) == \
        "relative weight of faults"
    assert m.strip_page_furniture("text continues here\n12", 12) == "text continues here"
    # numbers inside prose are untouched
    inside = "15 U.S.C. 1001 applies to a real sentence spanning 2026 words"
    assert m.strip_page_furniture(inside, 7) == inside
    # a rule/statute number at the start of a page is NOT a page number: the previous page ended a
    # sentence, so the number cannot be a marker glued to a wrapped sentence
    assert m.strip_page_furniture("15 U.S.C. 1001 applies", 15, None,
                                  prev_tail="shall be fined.") == "15 U.S.C. 1001 applies"
    # ...and when the previous page DOES end mid-sentence, a leading number EQUAL TO THE PAGE NUMBER
    # is the page marker pypdf glued to the wrapped sentence (this is the real §3.6.2 break:
    # page 11 ends "... and the", page 12's text begins "12 relative weight of faults ...")
    assert m.strip_page_furniture("12 relative weight of faults", 12, None,
                                  prev_tail="and the") == "relative weight of faults"
    # a leading number that is not this page's number is never touched
    assert m.strip_page_furniture("12 relative weight", 7, None,
                                  prev_tail="and the") == "12 relative weight"


def test_every_quoted_sentence_matches_the_committed_extraction():
    """Offline check that the committed report and the committed text tell the same story.

    The PDF is only reachable from a runner, so this reads the text the workflow committed.  Two
    directions matter, and both are asserted:

      * every quote the report calls a MATCH really is present in the committed text (a report that
        claims a verbatim match must be checkable by a reader with the evidence in hand);
      * every quote it calls a MISS really is absent (so a silent "not found" cannot hide a quote
        that the text does contain - which is exactly what the §3.6.2 page-number defect was).

    A report written before the page-furniture strip cannot be read this way, so it is skipped; the
    workflow refreshes it.
    """
    m = _mod()
    report = json.loads((ROOT / "data/evidence/rules_quotes.json").read_text())
    if not report.get("page_furniture_stripped"):
        import pytest
        pytest.skip("committed extraction predates the page-furniture strip")
    text_path = ROOT / (report.get("extracted_text") or {}).get("path", "")
    if not text_path.exists():
        import pytest
        pytest.skip("extracted text not committed in this checkout")
    text = text_path.read_text(encoding="utf-8")

    claimed = {q["id"]: bool(q["exact_match"]) for q in report["quotes"]}
    present = {q[0]: (m.norm(q[2]) in text) for q in m.QUOTES}
    wrong = {k: ("claimed match, absent from text" if claimed[k] else
                 "claimed miss, present in text") for k in claimed if claimed[k] != present[k]}
    assert not wrong, f"report and committed text disagree: {wrong}"
    assert report["summary"]["n_found"] == sum(claimed.values()), \
        "the report's own summary contradicts its rows"
    if report["summary"]["all_found"]:
        assert all(present.values()), "all_found is claimed but a quote is absent"
        # ...and if the strip was used, say so: the removals are listed in the report for review
        for r in report.get("page_furniture_removed") or []:
            assert set(r) == {"page", "removed", "why"}, "a removal must be explainable"
