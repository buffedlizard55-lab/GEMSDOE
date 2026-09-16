#!/usr/bin/env python3
"""Machine-verify the official-rules quotations this project's strategy rests on.

WHY: the single most consequential fact in this competition is *what the metric is computed
against*.  Everything else - architecture, loss, shaping, ensembling - is downstream of it.
A quoted sentence in a PDF is exactly the kind of claim that turns into folklore if nobody
re-checks it, so this script does not paraphrase: it extracts the text of the canonical
official rules and asserts each quoted sentence is present *verbatim* (after normalising
whitespace and unicode punctuation, because PDF extraction re-wraps lines and turns ' into ').

Source of truth (canonical, sponsor-hosted):
    https://docs.nlr.gov/docs/fy26osti/96647.pdf
The competition site links the same document through its rules page:
    https://www.drivendata.org/competitions/306/competition-doe-gems/rules/
    -> https://www.herox.com/GEMSPrize/resource/2274

Usage:
    python scripts/verify_rules_quotes.py --pdf data/GEMS_96647.pdf --out data/evidence/rules_quotes.json
    python scripts/verify_rules_quotes.py --url https://docs.nlr.gov/docs/fy26osti/96647.pdf --out ...
    python scripts/verify_rules_quotes.py --text some_extracted.txt     # offline test

Exit code is non-zero if any quote is missing, so it can gate CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

CANONICAL_URL = "https://docs.nlr.gov/docs/fy26osti/96647.pdf"
RULES_PAGE = "https://www.drivendata.org/competitions/306/competition-doe-gems/rules/"
PROBLEM_PAGE = "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/"

# (id, section, quote, why it matters for this project)
QUOTES: list[tuple[str, str, str, str]] = [
    ("phase1_target", "§1.1",
     "In Phase 1, submissions will be evaluated against a privately withheld subset of the "
     "original new fault dataset compiled by expert reviewers.",
     "Phase 1 is scored on NEW faults, not on the existing database we can download."),
    ("phase2_target", "§1.1",
     "Submissions will be reevaluated against the full, revised new fault dataset using the "
     "same distance-weighted Tversky index.",
     "Phase 2 (the $250k pool) is still scored on the NEW fault dataset - not on the public "
     "USGS catalogue. Copying the known-fault raster is therefore a trap in BOTH phases."),
    ("phase2_eligibility", "§1.1",
     "All Phase 1 competitors will be eligible to compete in Phase 2 and will be automatically "
     "submitted for consideration.",
     "No need to place top-5 in Phase 1 to win Phase 2, and one submission is used for both."),
    ("training_labels_source", "§3.3",
     "The training labels contain existing fault data at 100-m resolution where positively "
     "labeled pixels indicate fault presence. These labels were obtained from the INGENIOUS "
     "project's Great Basin Regional Dataset Compilation.",
     "The labels we train on are the EXISTING fault catalogue; the labels we are scored on are "
     "a different, newer population. Training labels and test labels are disjoint universes."),
    ("submit_new_labels", "§3.5",
     "Submissions will be automatically evaluated using a distance-weighted Tversky index "
     "against newly created fault labels, as described on the competition website.",
     "Third, independent statement of the same fact (problem page + rules §1.1 + §3.5)."),
    ("one_submission_two_rounds", "§3.5",
     "Before the end of the competition, you must choose only one submission for evaluation "
     "across both prize rounds.",
     "One raster must serve both rounds, so it cannot be tuned for a single phase."),
    ("blind_private_selection", "§3.6.2",
     "You must choose only one submission to use for scoring across both prize rounds, and you "
     "must make your decision without knowledge of your scores on the private test set.",
     "Private-set scores are hidden; the public leaderboard is a partial, differently-weighted "
     "sample, so it cannot be over-fitted."),
    ("weekly_limit", "§3.2",
     "You can make multiple submissions, subject to the limits specified on the competition "
     "website (three submissions per week).",
     "Three scored submissions per week is the only unbiased feedback channel available."),
    ("dem_is_a_feature", "§2",
     "In addition, the feature data also contain U.S. Geological Survey (USGS) Digital "
     "Elevation Model (DEM) elevation data at 1-m resolution.",
     "1 m DEM derivatives are explicitly part of the intended feature data (they are not yet "
     "used by our training configs)."),
    ("ai_disclosure", "§3.2",
     "you must indicate in the narrative (not included in the word count) the extent to which, "
     "if any, you used generative AI technology",
     "Binding disclosure requirement: this project uses AI agents, and the final narrative must "
     "say so."),
    ("code_assets", "§3.2",
     "The solution assets must contain a description of the resources required to build and "
     "run the solution, and they should be able to sufficiently reproduce the winning results "
     "and generate predictions on new data samples.",
     "A leaderboard-winning entry must ship reproducible code + resource documentation, which "
     "is what this repository is for."),
]


def _squash(s: str) -> str:
    """Remove ALL whitespace.  PDF text extraction re-wraps lines and, across page breaks, splits
    words ("...generate prediction" | "s on new data samples").  Comparing the whitespace-free form
    makes verbatim checking immune to layout without accepting paraphrases."""
    return re.sub(r"\s+", "", s)


def _norm(s: str) -> str:
    """Normalise for verbatim comparison: unicode punctuation, hyphenation, whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
          .replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
          .replace("\u00a0", " "))
    s = re.sub(r"-\s*\n\s*", "", s)          # de-hyphenate line breaks
    s = re.sub(r"\s+", " ", s)               # collapse newlines/indentation
    return s.strip()


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:                                       # pragma: no cover
        try:
            from PyPDF2 import PdfReader                       # type: ignore
        except ImportError:
            raise SystemExit("need pypdf (pip install pypdf) to read the PDF")
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="local PDF of the official rules")
    src.add_argument("--url", help=f"canonical URL (default {CANONICAL_URL})")
    src.add_argument("--text", help="already-extracted text (offline)")
    ap.add_argument("--out", default="data/evidence/rules_quotes.json")
    args = ap.parse_args()

    url = args.url or CANONICAL_URL
    pdf_sha = None
    source = {}
    if args.text:
        text = Path(args.text).read_text(errors="replace")
        source = dict(kind="text", path=str(args.text))
    else:
        if args.pdf:
            pdf = Path(args.pdf)
            source = dict(kind="file", path=str(pdf), bytes=pdf.stat().st_size)
        else:
            tmp = Path("/tmp/_rules.pdf")
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, tmp)
            pdf = tmp
            source = dict(kind="url", url=url)
        pdf_sha = sha256_file(pdf)
        source.update(sha256=pdf_sha, bytes=pdf.stat().st_size)
        text = extract_pdf_text(pdf)

    norm_doc = _norm(text)
    squash_doc = _squash(norm_doc)
    results = []
    n_ok = 0
    for qid, section, quote, why in QUOTES:
        nq = _norm(quote)
        ok = (nq in norm_doc) or (_squash(nq) in squash_doc)
        how = "substring" if nq in norm_doc else ("whitespace-insensitive" if ok else "none")
        # token-overlap fallback so a MISSING exact match reports how close it got
        toks = set(nq.split())
        best = 0.0
        if not ok:
            words = norm_doc.split()
            n = len(nq.split())
            for i in range(0, max(1, len(words) - n), 4):
                cand = set(words[i:i + n])
                if cand:
                    best = max(best, len(toks & cand) / len(toks))
        results.append(dict(id=qid, section=section, quote=quote, why=why,
                            exact_match=bool(ok), matched_as=how,
                            best_token_overlap=round(best, 4)))
        n_ok += int(ok)
        print(f"{'OK   ' if ok else 'MISS '} {qid:<26} {section:<6} "
              f"overlap={best:.3f}  {quote[:60]}...")

    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/verify_rules_quotes.py",
        document=dict(title="Geologic Enhanced Mapping System (GEMS) Prize Official Rules",
                      canonical_url=CANONICAL_URL, retrieved_via=source,
                      competition_rules_page=RULES_PAGE, problem_page=PROBLEM_PAGE),
        verification=dict(quotes_checked=len(QUOTES), exact_matches=n_ok,
                          all_verified=bool(n_ok == len(QUOTES)),
                          method="unicode NFKC + de-hyphenation + whitespace collapse, then a "
                                 "substring test; a second whitespace-free comparison absorbs the "
                                 "word splits that PDF page breaks introduce. Paraphrase is never "
                                 "accepted - the quoted characters must all be present, in order."),
        quotes=results,
        conclusion=("Both prize phases score the NEW fault dataset, which is disjoint from the "
                    "existing-fault labels the competition distributes for training. Route all "
                    "model selection through sources that do not assume otherwise.")
        if n_ok == len(QUOTES) else
        ("VERIFICATION INCOMPLETE - do not rely on the quotes above until every row is verified."),
    )
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1))
    print(f"\n{n_ok}/{len(QUOTES)} quotes verified verbatim -> {op}")
    return 0 if n_ok == len(QUOTES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
