#!/usr/bin/env python3
"""Extract the 1 m DEM download links from the competition's PDF and VERIFY each one.

The competition distributes the DEM link list as `1m_DEM_links.csv` on the data tab;
the public Dropbox mirror is a PDF print of the same JSON
("Digital elevation model links JSON.pdf").  PDF printing wraps and hyphenates long
URLs, so naive text extraction produces broken hosts (`prdtnm`, `prd.tnm`, `prd- tnm`)
— an irregularity recorded by an earlier session.  This script therefore:

  1. pulls raw text out of the PDF (pypdf),
  2. reassembles URLs across line breaks / soft hyphens,
  3. repairs the known host manglings back to the single real host,
  4. de-duplicates,
  5. HTTP HEADs every unique URL and records the status, size and ETag.

Only URLs that return HTTP 200 are written to the `verified` list, so nothing
downstream can consume a hallucinated or mistyped link.  Runs on the GitHub runner
(the sandbox cannot reach S3).

Usage: python scripts/extract_dem_links.py --pdf data/Digital-elevation-model-links-JSON.pdf \
         --out data/dem_links.json --evidence data/evidence
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

# The real USGS 3DEP distribution bucket. Verified host for all tiles.
CANONICAL_HOST = "prd-tnm.s3.amazonaws.com"
# Manglings observed in the PDF print of the JSON.
HOST_FIXES = [
    (r"prd\s*-?\s*tnm\s*\.\s*s3\s*\.\s*amazonaws\s*\.\s*com", CANONICAL_HOST),
    (r"prdtnm\.s3\.amazonaws\.com", CANONICAL_HOST),
    (r"prd\.tnm\.s3\.amazonaws\.com", CANONICAL_HOST),
]
URL_RE = re.compile(r"https?://[^\s\"'<>,\]\}]+", re.I)


def pdf_text(pdf: Path) -> str:
    from pypdf import PdfReader

    return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf)).pages)


def reassemble(text: str) -> str:
    """Undo PDF line wrapping inside URLs.

    A URL broken across lines shows up as '...tile' \n '_x_y.tif'. Any newline that is
    not followed by whitespace/quote/brace is treated as a wrap and removed. Soft
    hyphens introduced by the printer are dropped too.
    """
    t = text.replace("\u00ad", "")               # soft hyphen
    t = re.sub(r"-\s*\n\s*", "", t)              # hyphen + newline inside a token
    t = re.sub(r"\n\s*(?=[\w%/._~+-])", "", t)   # wrap directly into a URL-ish char
    for pat, rep in HOST_FIXES:
        t = re.sub(pat, rep, t, flags=re.I)
    return t


def head(url: str, timeout: int = 45) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gems-prize-audit"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {
                "status": r.status,
                "content_length": int(r.headers.get("Content-Length") or 0),
                "content_type": r.headers.get("Content-Type"),
                "etag": (r.headers.get("ETag") or "").strip('"'),
                "last_modified": r.headers.get("Last-Modified"),
            }
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": f"HTTPError {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"status": None, "error": repr(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="data/dem_links.json")
    ap.add_argument("--evidence", default="data/evidence")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    pdf = Path(a.pdf)
    if not pdf.exists():
        print(f"missing PDF: {pdf}")
        return 1

    raw = pdf_text(pdf)
    fixed = reassemble(raw)
    found = URL_RE.findall(fixed)
    # normalise trailing punctuation left over from the JSON print
    found = [u.rstrip('.,;"\'') for u in found]
    tiles = [u for u in found if u.lower().endswith((".tif", ".tiff"))]
    uniq = sorted(set(tiles))

    print(f"raw text {len(raw)} chars; {len(found)} URLs; {len(tiles)} tif; {len(uniq)} unique")

    records = []
    for u in uniq:
        rec = {"url": u, "filename": u.rsplit("/", 1)[-1]}
        if not a.no_verify:
            rec["head"] = head(u)
            st = rec["head"].get("status")
            mb = rec["head"].get("content_length", 0) / 1e6
            print(f"  {st}  {mb:8.1f} MB  {rec['filename']}")
        records.append(rec)

    verified = [r for r in records if r.get("head", {}).get("status") == 200]
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    payload = {
        "generated_utc": stamp,
        "source_pdf": pdf.name,
        "source_pdf_note": (
            "Dropbox mirror of the competition data-tab file '1m_DEM_links.csv', printed to PDF. "
            "URLs reassembled across PDF line wraps; known host manglings repaired to "
            f"{CANONICAL_HOST}."
        ),
        "n_urls_in_pdf": len(tiles),
        "n_unique": len(uniq),
        "n_verified_200": len(verified),
        "total_bytes_verified": sum(r["head"].get("content_length", 0) for r in verified),
        "records": records,
        "verified": [r["url"] for r in verified],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2) + "\n")

    ev = Path(a.evidence)
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "dem_links_raw_text.txt").write_text(raw[:200000])
    print(f"wrote {a.out}: {len(verified)}/{len(uniq)} verified HTTP 200, "
          f"{payload['total_bytes_verified']/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
