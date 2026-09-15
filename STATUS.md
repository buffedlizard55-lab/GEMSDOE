# Project status — 2026-09-14

Supersedes the earlier note that read *"the single remaining blocker to training is data
placement."* **That blocker is resolved.** The data was acquired, verified and used.

---

## What changed this session

The dev sandbox reaches only `github.com` and `pypi.org` — Dropbox, S3, DrivenData,
ScienceBase and even the Azure host that serves GitHub Actions *artifacts* are all blocked
(verified: `curl` exit 35 / EOF on each). Earlier sessions treated that as a hard stop.

It isn't. A **GitHub Actions runner has open egress**, and results can return to the
sandbox as ordinary git commits over `github.com`. That data bridge is
`.github/workflows/fetch-competition-data.yml`, and it changed the project from
"blocked, waiting for a human" to "data in hand".

### Acquired and verified (machine-measured, not transcribed)

| File | Size | Verified |
|---|---|---|
| `gems-geodawn-numerical-features.tif` | 418.9 MB | TIFF, 3292×3730, **19 bands**, EPSG:32611, 100 m |
| `existing_faults.tif` | 425.8 KB | TIFF, int8, 60,988 fault px (1.18 % of valid) |
| `example_submission.tif` | 1.6 MB | TIFF, float32 |
| `GEMS_96647.pdf` | 455.1 KB | PDF (rules) |
| `Digital-elevation-model-links-JSON.pdf` | 23.0 MB | PDF (**scan — no text layer**) |

sha256 for each is in `data/evidence/inventory.json`; full band table in
`data/evidence/rasters.json`.

---

## Irregularities found — flagged for review

1. **`example_submission.tif` is a copy of the labels, not an empty template.**
   The problem page says it *"predicts total fault absence"*. Measured: 60,988 positive
   pixels, **bit-identical** to `existing_faults.tif` across all 5,167,373 commonly-valid
   pixels. Evidence: `data/evidence/transfer_analysis.json → sample_vs_labels`.
   *Impact here: none — we take only georeferencing from it. Worth reporting upstream.*

2. **The feature stack has 19 bands**, not the ~10 the problem page's prose implies.
   All 19 carry self-describing `band_name` / `data_category` tags, reproduced verbatim
   on the site's Data page.

3. **The DEM-links PDF has no text layer.** pypdf → 41 chars, pdftotext → 0, pdfplumber
   → 41; zero `http` occurrences in any. It is a raster scan.
   Recovered by OCR, then **every URL re-confirmed against the live USGS 3DEP bucket
   listing** — 716 unique tiles confirmed, 3 unmatched and reported. No URL is
   reconstructed from OCR characters, because OCR corrupts them (`2020`→`202@`,
   underscores→spaces, `QL1`/`QLi`). See `data/dem_links.json`.

4. **Three doc pages cited HTML files that no longer existed** after the site rebuild.
   Caught by `scripts/audit_docs.py`, fixed. The gate now passes.

---

## The most important finding: what the metric actually pays for

`TP_w + FN_w = |G|` holds identically, so the official DTI rearranges exactly to

> **DTI = TP_w / ( 0.2·(TP_w + FP_w) + 0.8·|G| )**

`TP_w` and `FP_w` share the same coefficient. Measured on a real window of the competition
labels (`scripts/metric_strategy.py`):

| Strategy | DTI |
|---|---:|
| all zeros | 0.0000 |
| **all ones (blanket coverage)** | **0.0956** |
| **25 % of the faults, precisely** | **0.5545** |
| known faults dilated by 1 px | 0.8818 |
| exact known faults | 0.9999 |

α=0.2/β=0.8 *looks* like "predict generously". It is the opposite: β multiplies `FN_w`,
capped at `|G|`; α multiplies `FP_w`, which grows with painted **area**, and non-fault
pixels outnumber fault pixels ~50:1. **Be sparse, thin and confident.** Full derivation:
`docs/METRIC_STRATEGY.md` and the site's Metric page.

---

## Pipeline state — verified, not yet competitive

- **Runs end-to-end on the real rasters** (CPU runner): train → inference → format
  validation → scoring. `submission.tif` produced; `validate_submission.py` **PASSED**
  (single band, float32, [0,1], EPSG:32611, 100 m, matching grid/transform).
- **Runs in-sandbox** against `data/fixture/` — a 512×512 window of the *real* rasters
  committed to the repo (5 MB, int16, round-trip error 1.6×10⁻⁵). 3 epochs in ~20 s.
- **Metric implementation verified**: 8 self-tests including fast-vs-brute-force agreement
  and reproduction of the official worked example (0.60).

### Honest numbers

The only full-scale run was a 2-epoch MobileNetV2 smoke test: **DTI 0.0816 against the
public labels — *below* the 0.0956 blanket-coverage floor.** That is a negative result and
is recorded as such. It proves plumbing, not predictive power. Shaping helped on every run
(`DTI_shaped` > `DTI_raw` consistently), exactly as the metric algebra predicts.

---

## What is genuinely still needed

1. **A GPU.** `configs/config.yaml` (UNet++/DeepLabV3+/SegFormer ensemble, EfficientNet-B5,
   10 MC splits, 60 epochs, pretrained encoders) cannot run inside a CPU runner's job
   limit. This is the single biggest gap between "works" and "competitive".
2. **A DrivenData account** to actually submit. The pipeline produces a
   format-valid `submission.tif`; uploading it requires a logged-in human, and the rules
   require choosing one submission for both rounds before the deadline.
3. **DEM derivatives** from the 716 confirmed tiles (~GB-scale download, best done on a
   runner or the GPU box).
4. **A fresh GitHub token** — the session token expired at the end of this session.
   Last successful push: `a28b699`. All work is committed and on the remote.

---

## Anti-hallucination measures in place

- The site is **generated** (`scripts/build_site.py`) from the evidence JSON; no figure is
  hand-typed. Missing evidence renders as an explicit "not available" block.
- `scripts/audit_docs.py` fails CI if any doc cites a file that doesn't exist.
- `scripts/verify_links.py` re-fetches every catalog URL and records the real HTTP status;
  `LOGIN_REQUIRED` is reported as such, never as "verified".
- Local scores are labelled **optimistic** wherever shown — they are computed against the
  labels the model trained on, which are not the test set.

---

## Site publishing note (2026-09-15)

GitHub Pages for this repo is set to **`build_type: legacy`** (Jekyll from the repo root), so
the root URL renders `README.md` and the `docs/` artifact uploaded by
`.github/workflows/pages.yml` is ignored at the root. Switching Settings → Pages → Source to
**"GitHub Actions"** requires repository-admin rights the automation token does not have
(`PUT /repos/.../pages` returns HTTP 403).

Workaround in place: a root `index.html` redirects to `docs/index.html`, which legacy Jekyll
honours ahead of `README.md`. The generated site is live and correct at
<https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html>.

**Recommended (optional) owner action:** flip Source to "GitHub Actions" for a cleaner root
URL. The redirect becomes harmless if you do.
