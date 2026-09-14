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

# OCR confusions seen in scanned URL listings. Applied ONLY inside the fixed, known
# path prefix of the 3DEP bucket - never to the tile filename, where a wrong repair
# would invent a tile that does not exist. Every resulting URL is HTTP-verified.
PATH_FIXES = [
    (r"StagedProduct[sS]", "StagedProducts"),
    (r"E1evation", "Elevation"),
    (r"e1evation", "elevation"),
    (r"TIFE", "TIFF"),
    (r"Tl[FE]F", "TIFF"),
    (r"//+", "/"),
]


def pdf_text(pdf: Path) -> str:
    """Text from the PDF, trying text-layer extractors first, then OCR.

    MEASURED on the competition file (runner, 2026-09-14): the 23 MB Dropbox print of
    `1m_DEM_links.csv` has NO text layer at all - pypdf 41 chars, pdfplumber 41 chars,
    pdftotext 0 chars, zero 'http' occurrences in any of them.  It is a raster scan.
    So OCR (tesseract at 300 DPI) is the only way to read it, and even then the output
    must be treated as untrusted until each URL is confirmed against the live S3 bucket
    - which `verify`/`head` below does.
    """
    results: dict[str, str] = {}

    def try_(name, fn):
        try:
            results[name] = fn() or ""
        except Exception as e:  # noqa: BLE001
            results[name] = ""
            print(f"  {name} failed: {e}")

    def _pypdf():
        from pypdf import PdfReader

        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf)).pages)

    def _pdftotext():
        import subprocess

        return subprocess.run(["pdftotext", "-layout", "-nopgbrk", str(pdf), "-"],
                              capture_output=True, text=True, timeout=1800).stdout

    def _pdfplumber():
        import pdfplumber

        with pdfplumber.open(str(pdf)) as doc:
            return "\n".join((pg.extract_text() or "") for pg in doc.pages)

    def _ocr():
        """Rasterise each page at 300 DPI and OCR it.

        `--psm 6` = assume a uniform block of text, which suits a printed JSON listing.
        The character whitelist is NOT restricted: a wrong guess would silently corrupt
        URLs, and every URL is HTTP-verified afterwards anyway.
        """
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["pdftoppm", "-r", "300", "-png", str(pdf), f"{td}/pg"],
                           check=True, timeout=3600)
            pages = sorted(Path(td).glob("pg*.png"))
            print(f"  OCR: {len(pages)} page images at 300 DPI")
            chunks = []
            for i, img in enumerate(pages, 1):
                out = subprocess.run(["tesseract", str(img), "stdout", "--psm", "6"],
                                     capture_output=True, text=True, timeout=600)
                chunks.append(out.stdout)
                if i % 10 == 0 or i == len(pages):
                    print(f"    OCR page {i}/{len(pages)}")
            return "\n".join(chunks)

    try_("pypdf", _pypdf)
    try_("pdftotext", _pdftotext)
    try_("pdfplumber", _pdfplumber)
    if max(v.lower().count("http") for v in results.values()) == 0:
        print("  no text layer found in any extractor -> falling back to OCR")
        try_("ocr_tesseract_300dpi", _ocr)

    for name, txt in results.items():
        print(f"  extractor {name}: {len(txt)} chars, {txt.lower().count('http')} 'http'")
    best = max(results, key=lambda k: (results[k].lower().count("http"), len(results[k])))
    print(f"  -> using {best}")
    globals()["_EXTRACTOR_USED"] = best
    globals()["_EXTRACTOR_STATS"] = {k: {"chars": len(v), "http": v.lower().count("http")}
                                     for k, v in results.items()}
    return results[best]


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
    t = t.replace("https:/" + "/", "https://")   # normalise after the // collapse below
    return t


def repair_path(url: str) -> str:
    """Repair OCR damage in the *fixed* part of a 3DEP URL, leaving the filename alone."""
    try:
        scheme, rest = url.split("://", 1)
        host, path = rest.split("/", 1)
    except ValueError:
        return url
    head, _, tail = path.rpartition("/")
    for pat, rep in PATH_FIXES:
        head = re.sub(pat, rep, head)
    return f"{scheme}://{host}/{head}/{tail}" if head else url


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
    found = [repair_path(u.rstrip('.,;"\'')) for u in found]
    tiles = [u for u in found if u.lower().endswith((".tif", ".tiff"))]
    uniq = sorted(set(tiles))

    print(f"raw text {len(raw)} chars; {len(found)} URLs; {len(tiles)} tif; {len(uniq)} unique")

    records = []
    if a.no_verify:
        records = [{"url": u, "filename": u.rsplit("/", 1)[-1]} for u in uniq]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=16) as ex:
            heads = list(ex.map(head, uniq))
        for u, h in zip(uniq, heads):
            rec = {"url": u, "filename": u.rsplit("/", 1)[-1], "head": h}
            print(f"  {h.get('status')}  {h.get('content_length', 0) / 1e6:8.1f} MB  {rec['filename']}")
            records.append(rec)

    verified = [r for r in records if r.get("head", {}).get("status") == 200]
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    payload = {
        "generated_utc": stamp,
        "source_pdf": pdf.name,
        "extractor_used": globals().get("_EXTRACTOR_USED"),
        "extractor_stats": globals().get("_EXTRACTOR_STATS"),
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
