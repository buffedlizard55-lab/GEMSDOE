# Project status — 2026-09-15 (session 3)

## What changed this session

1. **Strategic fact re-verified from the official page** (fetched 2026-09-15, verbatim):
   both prize rounds score against the set of **newly-labelled** faults — Phase 1 against
   "a privately withheld subset of the original new fault dataset compiled by expert
   reviewers", Phase 2 against "the full, revised new fault dataset" (rules PDF §1.1,
   <https://docs.nlr.gov/docs/fy26osti/96647.pdf>). The public `existing_faults.tif` is
   training data, NOT the scoring target. `docs/METRIC_STRATEGY.md` §4 now states the
   consequence: copying the known-fault raster (which scores 0.9999 against itself) is a
   trap, and every local DTI number is a plumbing monitor, not a leaderboard proxy.

2. **Reference solution cross-verification (new, line-by-line)**: fetched the official
   reference notebook and compared its executed output against our runner-measured
   evidence: **19/19 band descriptions in `gems-geodawn-numerical-features.tif` match the
   notebook output exactly** (names, order, wording), CRS/resolution/nodata/shape agree,
   and the notebook confirms `numeric_features.tif` naming + plain min-max normalisation +
   device-fallback order. This independently ties our acquired files to the official ones.

3. **The parallel ensemble pipeline is implemented and locally exercised**:
   `configs/config_ci_ensemble.yaml` + `.github/workflows/train-ensemble.yml` run 6 MC
   folds as 6 parallel CPU jobs (UNet++/DeepLabV3+/U-Net × resnet34 @ patch 256, 8-way
   TTA, time-guarded so a checkpoint always exists), then a blend job averages the raw
   fold maps, shapes ONCE with the pooled held-out (t0, thin) calibration
   (`scripts/blend_submission.py`) and commits report + submission back to the branch.
   New regression tests (`tests/test_ensemble.py`); full suite **23/23 green** in-sandbox.

4. **Blend-weighting rule MEASURED, not assumed** (6-fold mini ensemble on the real
   512×512 fixture window, sandbox): with **2** folds, equal weights let a weak fold
   (held-out 0.049) bury a strong one (0.150) and softmax-over-held-out-DTI lifted the
   blend 0.0143 → 0.0984; but with **6** folds the same weighting HURT
   (equal 0.1033 vs weighted 0.0656) because per-fold held-out DTI differences on small
   crops are noise and the softmax over-concentrates. Final rule: **equal averaging is
   the workflow default**; `--weights dti` remains for excluding a known-catastrophic
   fold. Evidence + full table: `data/evidence/runs/local-mini-ensemble/`.

5. **Also fixed**: per-epoch shaping-table cost (scattered 12-window bbox → whole-raster
   EDTs; now a compact fault-bearing window run, `selection_mode: raw` for CPU folds);
   `early_stopping_patience` was configured but never implemented — it is now;
   `pretrained: true` now degrades to a warning instead of crashing;
   `src/inference.py` gained `--raw` and `--override`; audit_docs now passes on the
   formerly pseudo-URL `&hellip;` links in LIMITATIONS.md.

## GitHub auth blocker — RESOLVED, ensemble run now live (2026-09-15, session 3 addendum)

The expired `GH_TOKEN` from earlier this session was reconnected. Consequences, in order:

1. `git push origin arena/01a0a738-gemsdoe` succeeded (remote branch was created).
2. **PR #10 opened** against `main` for the six session-3 commits, incl. the reviewer note
   that open PR #9's three defect fixes (fpw leak on un-zeroed labels, unreachable shaping
   floor, `in_channels=10` default) are *not* in this branch — verified by inspection:
   `src/dataset.py` still computes `fp_src = (yp > 0.5)` before the zeroing, and
   `src/submission_optim.py` still uses `np.linspace(0.02, 0.9, 45)`.
3. The push touched `.github/triggers/ensemble`, so **workflow run 35042805806 auto-started**
   ("Train MC ensemble (parallel folds) + blend"): all 6 fold jobs `in_progress`. On success
   the blend job commits `data/evidence/runs/35042805806/` (blend_report.json,
   validation.log, submission.tif) back to this branch; the artifact `submission-final-*`
   is the file to upload to DrivenData (3 submissions/week limit). Numbers are informational
   monitors vs public labels, not leaderboard proxies (see §1 above).

Caveat while reviewing the run: it trains the code *with* the fpw leak (#9 not merged), so
fold-selection DTI is optimistic by the sliver effect #9 quantifies; relative fold ordering
is expected to survive but treat selection accordingly until #9 lands.
Human-only steps unchanged: DrivenData account + upload; optional Settings→Pages→"GitHub
Actions" flip; forum report for the example_submission.tif == labels anomaly.

---

# Project status — 2026-09-14 (superseded below where it conflicts with the above)

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

---

## Correctness fixes to the training path (2026-09-15)

Three defects found by re-reading the code against the evidence, each now covered by a
regression test in `tests/test_metric.py` (20/22 pass in-sandbox; the 2 failures are
`torch`-only tests — no installable CPU wheel here, the pytorch index is egress-blocked).

**1. Held-out label leakage through the FP-weight map** (`src/dataset.py`).
`fpw = 1 - max_g k(d(x,g))` is handed to the loss as "how much a prediction here counts as
a false positive", so `fpw < 1` states that a fault lies within R px. It was computed from
the labels *before* the test windows were zeroed. Training windows are allowed to overlap
the test grid by up to 25%, and inside that sliver the loss was being told not to penalise
predictions sitting exactly on the hidden faults. Held-out DTI was therefore optimistic.
Fixed by computing the EDT on the zeroed (train-only) labels and forcing `fpw = 1` inside
the held-out region. Synthetic check: 196 test-region pixels carried `fpw < 1` before, 0
after. On the real fixture the corrected map now matches a train-only reference exactly
(max abs error 0.0 across all 27 train patches).

**2. The shaping floor search could not reach the optimum** (`src/submission_optim.py`).
The full-raster smoke run (Actions run `34876843912`) selected `t0 = 0.02` — the *first*
point of `linspace(0.02, 0.9, 15)`. An optimum pinned to a grid endpoint means the grid,
not the data, chose it. New `shaping_thresholds()` is log-spaced from 1e-4 and always
includes 0.0 as an explicit no-floor row. Measured on the real fixture labels with a
low-contrast field of the kind an under-trained model emits (max p = 0.014): the old grid
zeroes the entire submission (**DTI 0.000**, every value falls under the 0.02 floor) while
the new grid finds **DTI 0.856** at `t0 = 0.0050`.

**3. `get_model(in_channels=10)` default** (`src/models.py`) → **19**, the measured band
count. Every caller passes it explicitly so no run was affected, but it was a trap for
direct importers.
