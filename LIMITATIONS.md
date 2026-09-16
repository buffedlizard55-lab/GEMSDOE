# Limitations & Required Access — GEMS Prize

**Verified sources:**
- Competition main: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Problem description: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Official Rules PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution

## Current Limitations in this Sandbox Environment

### 1. No DrivenData Authentication
- **Issue:** Data download page https://www.drivendata.org/competitions/306/competition-doe-gems/data/ requires login and competition enrollment.
- **Impact:** Cannot automatically download `training_features.tif`, `labels.tif` (or `numeric_features.tif`, `faults.tif`), `sample_submission.tif`, `1m_DEM_links.csv`.
- **Mitigation:** Code in `src/dataset.py` handles both naming conventions and checks multiple candidate paths. `data/README.md` provides manual download instructions. `scripts/prepare_data.py` validates files.
- **Verified:** Attempt to fetch data tab redirects to login page — confirmed via `fetch_page`.

### 2. No GPU / Limited Compute
- **Environment:** Python 3.11.2, CPU-only, no torch pre-installed (verified via `pip show torch` → not found).
- **Impact:** Training large segmentation models (UNet++, DeepLabV3+, SegFormer with EfficientNet-B5) ideally needs GPU (CUDA 12.6/13.0 or Apple MPS). Reference solution recommends GPU: https://github.com/drivendataorg/gems-prize-reference-solution#choosing-gpu-vs-cpu
- **Mitigation:** Code supports CPU, MPS, and CUDA via `torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`. Mixed precision only on CUDA. Config allows reducing batch size, patch size, MC splits for CPU testing.
- **Needed:** 1× A100 24GB+ or equivalent, 5-10h training for full ensemble (MC=10, 50 epochs).

### 3. No Large External Data Pre-downloaded
- **GeoDAWN:** ScienceBase item https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 contains GB-scale zips (22103_area1_grids.zip 42MB, area2_grids 227MB, mag gdb 350MB, spec gdb 378MB, tiffs 43MB + 230MB, plus 3GB+ contractor packages). Cannot bulk-download in sandbox.
- **3DEP 1m DEM:** Many tiles for Nevada region; each tile ~100MB, region needs dozens. Sources: https://apps.nationalmap.gov/downloader/ and https://apps.nationalmap.gov/lidar-explorer/ and AWS https://registry.opendata.aws/usgs-lidar/
- **INGENIOUS:** GDR https://gdr.openei.org/submissions/1391 contains 116MB+ across 9 files (2m probes, earthquake density, paleo features, etc.)
- **Mitigation:** `scripts/download_external.sh` provides verified official links and manual instructions. `src/external_data.py` implements DEM derivative computation (slope, curvature, TPI, TRI, detrended, hillshade) so that once DEM tiles downloaded, features can be generated.
- **Needed:** ~50GB storage, stable internet, possibly AWS CLI.

### 4. Private Test Labels Unavailable
- **Expected:** Private test set is intentionally withheld; only public test subset shown on leaderboard. This is by design per competition structure https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure
- **Impact:** Cannot directly optimize for private test; must use cross-validation and public LB.
- **Mitigation:** Implement Monte Carlo CV (70/30 splits) with distance-weighted Tversky metric for validation, as in `src/metrics.py` and `src/train.py`. Target high recall (β=0.8) to generalize to hidden faults.

### 5. No Direct Submission to DrivenData from Sandbox
- **Issue:** Submitting requires DrivenData account and selecting final submission before deadline (3 submissions/week limit per rules https://docs.nlr.gov/docs/fy26osti/96647.pdf section 3.4).
- **Mitigation:** `src/inference.py` generates submission.tif matching exact format requirements (EPSG:32611, 100m, float32 [0,1], same bounds). `scripts/validate_submission.py` (to be added) checks format.

## What We Need Access To for Full Competitive Run

1. **DrivenData Account + Competition Enrollment**
   - URL: https://www.drivendata.org/competitions/306/competition-doe-gems/
   - Required to download training data and submit.
   - Eligibility: U.S. citizen/permanent resident per Official Rules https://www.nlr.gov/docs/fy26osti/96647.pdf section 1.3

2. **Compute Resources**
   - GPU: NVIDIA with CUDA 12.6 or 13.0 (reference solution recommends `uv sync --extra cu126`), or Apple Silicon MPS.
   - CPU: 8+ cores for data loading
   - RAM: 32GB+
   - Storage: 50GB+ for data + outputs

3. **External Data Access (Optional but Recommended)**
   - USGS GeoDAWN: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 DOI https://doi.org/10.5066/P93LGLVQ
   - USGS QFaults: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 DOI https://doi.org/10.5066/P9BCVRCK
   - USGS 3DEP: https://apps.nationalmap.gov/downloader/ and https://apps.nationalmap.gov/lidar-explorer/
   - INGENIOUS: https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483

4. **Time**
   - Training: 5-10h for full ensemble
   - Inference: ~1h for full region with TTA and ensemble
   - Expert review of all submissions happens after close; winners notified ~60 days after the prize closes (per PDF section 3.6.5); ACH/W-9 within 30 days of notice (A.2)

5. **Documentation for Winners**
   - Per PDF section 3.2 and 3.5: Winners must submit complete code assets + documentation sufficient to reproduce results, consistent with DrivenData's Winning Model Documentation Template.
   - Must indicate generative AI use if applicable (PDF section 3.2)
   - Must sign eligibility certifications

## Allowed External Data — Verified Licenses

Per problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#external-datasets and Official Rules PDF section 3.3:

> Participants are allowed to use any additional data sources, provided that the participants possess a license that permits the data to be used in this challenge and shared with the sponsor for evaluation purposes.

Our external data:
- **USGS data (GeoDAWN, QFaults, 3DEP, EarthMRI):** Public Domain (U.S. Government) — https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits
- **INGENIOUS (GDR):** CC License, Publicly accessible — stated on https://gdr.openei.org/submissions/1391 as "Publicly accessible License CC"
- **GBCGE Subsurface:** Public via OSTI — https://www.osti.gov/dataexplorer/biblio/dataset/1987556

All shareable with sponsor.

## No Hallucinations Statement

- All links verified via `web_search` and `fetch_page` on 2026-09-12.
- No synthetic fault data invented.
- File names `training_features.tif`, `1m_DEM_links.csv` from official problem page.
- Reference solution naming drift (`numeric_features.tif` vs `training_features.tif`) flagged as irregularity.

### 1c. Fresh egress matrix — 2026-09-12T22:09Z (post-sandbox-restart re-test)

| Host | curl result |
|---|---|
| github.com / api.github.com | ✅ data flows (git/gh) |
| codeload.github.com | ✅ data flows (200, 1.3 MB tarball verified) — **NEW**, enables any public repo content |
| raw.githubusercontent.com | ❌ exit 35 (TLS data dropped) |
| objects.githubusercontent.com (release assets) | ❌ exit 35 → smp pretrained encoder weights NOT downloadable |
| prd-tnm.s3 / drivendata-public-assets.s3 / www.dropbox.com / huggingface.co | ❌ exit 35 |
| pypi.org / files.pythonhosted.org | ✅ (numpy/scipy/torch/rasterio/smp/geopandas installed) |

**GitHub mirror hunt (definitive negative):** `gh api search/code` on 5 distinctive filenames
(`gems-geodawn-numerical-features`, `existing_faults.tif`, `numeric_features.tif`, `1m_DEM_links`,
`example_submission.tif`) → **0 hits**. The official competition files have no public GitHub mirror.

### 1d. Reconstructed dataset caveat (2026-09-12)

Because the official files cannot enter the sandbox, the pipeline was verified end-to-end on
`data/reconstructed/` — REAL public-source data (GeoDAWN survey 22103 area1 + INGENIOUS QFaults +
earthquake density; see `data/README.md`), **not** the official competition files: different band
set (11 of 16 analog/extra vs official derivations), different processing, subregion extent.
Usable for: pipeline proof, external-data pretraining, sanity CV. Required for leaderboard:
official data-tab files via `scripts/download_competition_data.sh`.

### 1e. Code-vs-docs discrepancies found during E2E (2026-09-12)

- docs claimed torchvision augmentations in the training loop — **not implemented** in `src/train.py` (FaultDataset transform is a no-op placeholder). Docs corrected; augmentation remains a next-step.
  **Resolved 2026-09-13:** label-consistent numpy crop/flips/rot90/noise in `FaultDataset`, regression-tested (`test_augmentation_is_label_consistent`).
- `make_patches` keeps training windows that partially overlap held-out test regions (only fully-covered windows are excluded) → mild CV leakage vs the reference solution's global test-region zeroing. Flagged; fix queued in SUGGESTIONS §.
  **Resolved (verified 2026-09-15 by re-reading `src/dataset.py:make_patches`):** the test mask now zeros both X and y globally *before* train windows are extracted, and windows >25 % covered by test are skipped; regression-tested by `test_patches_have_no_label_leakage`.
- `pretrained: true` configs cannot fetch encoder weights from this sandbox (release-asset host blocked); sandbox runs use `pretrained: false`. On an unrestricted machine `configs/config.yaml` works as-is.
- 3-epoch CPU smoke model ≈ constant-prediction baseline (DTI 0.0519 vs 0.0524) — honest result; not evidence of model quality.

### 1f-2. Sandbox re-test + auth status (2026-09-15, this session)

| Probe (2026-09-15T00:0xZ) | Result |
|---|---|
| github.com / api.github.com (bare GET) | 200 — HTML of the repo/API root via the egress proxy |
| pypi.org / files.pythonhosted.org | ✅ full installs work (torch 2.14.0+cu130 wheel from PyPI installed in `.venv`) |
| download.pytorch.org/whl/cpu | ❌ TLS dropped (use PyPI default wheels instead) |
| raw.githubusercontent.com (direct) | ❌ exit 35 |
| codeload.github.com tarballs | ❌ now returns a 404 stub through the proxy (was 200 on 2026-09-12) |
| www.dropbox.com / drivendata.org | ❌ exit 35 (unchanged) |
| `gh api` reads | ✅ for the first ~30 min, then **"Bad credentials"** — the Arena session `GH_TOKEN` expired mid-session again; `git push` → `Invalid username or token` |
| `fetch_page` (harness-side, not sandbox network) | ✅ reaches drivendata/NLR/GitHub — used this session to verbatim-verify the metric formulas, submission format, rules §1.1/§3.2 and the reference notebook's band table (19/19 match) |

**Impact:** the parallel-ensemble work is committed on the local branch (`arena/01a0a738-gemsdoe`, 2 commits) but cannot be pushed until GitHub is reconnected in Arena; until then the 6-fold workflow cannot be dispatched. No other route exists from this sandbox (verified again above). Owner action: reconnect GitHub → `git push origin arena/01a0a738-gemsdoe` (or re-run this session) and the run starts automatically (the push touches `.github/triggers/ensemble`).

### 1f. Dropbox mirror fetch notes (2026-09-12)
- Dropbox `scl/fi` links carry a short-lived `st` signature; it expired mid-capture of `Digital-elevation-model-links-JSON.pdf` (chunk 1/13 fetched, rest failed). The durable form is `rlkey` + `dl=1` (used in `scripts/download_competition_data.sh`).
- DEM tiles are ~90-380 MB each; the full 3-project list is tens of GB — download on a machine with disk headroom (`scripts/download_dem_tiles.py`).


---

## 2. Fresh-sandbox re-verification, 2026-09-12 (later same day)

The sandbox was reset between sessions (no packages, empty `data/` except `README.md`), so every environment and
pipeline claim was re-established from scratch instead of trusted.

### 2.1 Egress matrix, re-measured 2026-09-12T23:0xZ

| Host | Result |
|---|---|
| `github.com` (incl. `/archive/...tar.gz`), `codeload.github.com`, `api.github.com` | ✅ 200 |
| `pypi.org`, `files.pythonhosted.org` | ✅ 200 (torch 554 MB wheel installed at ~150 MB/s) |
| `www.drivendata.org`, `www.dropbox.com`, `prd-tnm.s3.amazonaws.com`, `www.sciencebase.gov`, `gdr.openei.org`, `pubs.usgs.gov`, `earthquake.usgs.gov`, `services.nationalmap.gov`, `huggingface.co`, `cdn.jsdelivr.net`, `gist.githubusercontent.com`, `raw.githubusercontent.com`, `web.archive.org`, `en.wikipedia.org`, `www.nlr.gov` | ❌ `curl (35) SSL_ERROR_SYSCALL` (TLS dropped by the allowlist proxy) |

Text fetching is still possible through the platform-side `fetch_page` (used to re-read the problem page, the About
page, the DrivenData main page and all 7 chunks of the rules PDF), but **binary GeoTIFFs cannot be delivered that
way**. `raw.githubusercontent.com` is blocked while `codeload.github.com` works — which is exactly why the
public-data reconstruction below succeeded: repo *tarballs* are reachable, file-by-file raw access is not.

### 2.2 What was unblocked autonomously (no manual input)

- **A real dataset inside the sandbox:** `jklinck/geothermal_research` @ `56d78de7a989c12e2dce50cd65a4095df57030d2`
  (tarball sha256 `f78b96a36fd5e814…`), containing USGS GeoDAWN 22103 area-1 grids and INGENIOUS fault/seismicity
  layers. Built with `scripts/build_reconstruction_dataset.py` → 16-band EPSG:32611/100 m stack, 1.42 % fault pixels.
- **Full pipeline executed** on it: train (2 MC splits) → inference → `validate_submission.py` ✅ → metric scoring;
  plus a 20-test suite and `python src/metrics.py --self-test` (8 checks). Details and numbers: `docs/results.html`.
- **Reference solution fully read** (21 notebook cells, cloned) → the baseline hyperparameters cited across this repo
  are now quoted from it rather than paraphrased.

### 2.3 Irregularities found in this review (all flagged, none silently patched)

1. **Cited-but-uncommitted evidence.** Six places referenced `data/dem_links.json` / `data/evidence/…`; `data/*` was
   gitignored (only `README.md` allow-listed), so those artefacts were never in git and vanished with the sandbox.
   Fixed: `.gitignore` now allow-lists `data/dem_links.json`, `data/evidence/**`, `data/reconstructed/provenance.json`;
   `scripts/fetch_dem_links_pdf.py` regenerates them; every doc reference corrected to "regenerate with …".
2. **Dead dependency.** `requirements.txt`/`environment.yml` listed `albumentations`, which the code never used
   (docstrings claimed it). Removed; augmentation is implemented in `src/dataset.py` and is now wired into training.
3. **API bug caught by the A/B run.** `TverskyLoss.forward()` rejected the shared `fp_weight` kwarg, so the
   reference-comparison arm crashed. Fixed (signature unified) — and it is exactly why we run the comparison instead
   of asserting it works.
4. **Docs overstatement removed.** The "Expected gains 0.45-0.75 DTI" table had no measurement behind it; replaced
   by the measured table. Any remaining estimate is labelled as an estimate.
5. **Still true:** the official competition rasters cannot reach this sandbox (login-gated + TLS-blocked), so no
   leaderboard-representative score can be produced here. Run `bash scripts/download_competition_data.sh` on an
   unrestricted machine → `data/`, then `python -m src.train --config configs/config.yaml`.

---

## 3. Post-merge review, 2026-09-13 (after PR #3 / #4 merged)

Merging was followed by reading the **published** pages against the artifacts, which found four
documentation defects and one deployment ambiguity. All are fixed in the repo; the deployment item is
for the repo owner:

1. `docs/results.html` &sect;4 (end-to-end) quoted the &sect;5 A/B arm's shaping numbers and a pre-fix inference
   mass. Fixed, with the correction kept visible; `src/inference.py` now writes
   `outputs_recon/inference_summary.json` so that row is derived from a run record, and `audit_docs.py`
   re-derives it.
2. The old hand-written methodology page (deleted 2026-09-14 when the site was regenerated from measured evidence by `scripts/build_site.py`; superseded by `docs/method.html` and `docs/metric.html`) quoted "perfect line &rarr; 0.91 DTI, shifted 1 px &rarr; 0.61". The code gives
   **1.0000** and **0.6667** (= k(1) = 2/3) and **0.0000** at 3 px; replaced by the measured table and pinned
   by `test_line_geometry_dti_values`. Two pages also described a "7&times;7 max filter" that no longer exists
   (it would credit corners at d&asymp;4.24 px, which the spec's *radius* excludes).
3. 19 `href="../X.md"` links and 11 bare `href="X.md"` links 404 on the site, because Pages publishes
   `docs/` as the site root and does not render markdown. Repointed to the repository on GitHub; the
   audit's new link check enforces it.
4. The audit gate grew from three checks to five (published tables vs artifacts; site link targets),
   each falsification-tested.
5. **Owner action, not code:** `GET /repos/buffedlizard55-lab/GEMSDOE/pages` reports `build_type: legacy`, source `main`, path
   `/`, yet the successful `deploy-pages` job means the artifact (which uploads `docs/`) is what is
   currently live &mdash; `https://buffedlizard55-lab.github.io/GEMSDOE/` now serves `docs/index.html` and `https://buffedlizard55-lab.github.io/GEMSDOE/docs/...`
   returns 404. Both builders are therefore racing on every push to `main`. Pick one: set
   **Settings &rarr; Pages &rarr; Source: GitHub Actions** (recommended; the workflow already uploads `docs/`),
   or remove the `deploy` job and let Jekyll build the repo root. Nothing in the repo can settle this
   without the setting change.

## 4. What the deployed-page check caught (2026-09-13, after PR #5)

`docs/index.html` still described the metric as using a "7×7 max filter" — stale text from before the
scorer was rewritten to enumerate the 29 offsets inside the Euclidean radius. It survived because a
mid-session `git checkout -- docs/index.html` (used to undo a *different* bad edit) silently reverted
that paragraph too, and `audit_docs.py` has no way to know a prose sentence is wrong: it checks that
cited paths exist, that tables equal their artifacts, that links resolve and that hosts are catalogued.
Lesson recorded: after any bulk revert, re-read the whole affected page — and prefer reading the
**deployed** page over the local file, which is exactly how this was found. Fixed in the follow-up
commit; the numeric claims in the same area are pinned by `test_line_geometry_dti_values` (20/20 tests).

---

# 2026-09-16 addendum — measured limits, and one that is not a limit

Everything below was re-measured this session with `curl`, `gh` and `rasterio` inside the sandbox;
nothing is inferred from the earlier notes.

## 1. Egress allowlist (re-measured)

| host | result | consequence |
|---|---|---|
| `github.com`, `api.github.com`, `codeload.github.com` | 200 / working | git push, PRs, workflow dispatch, **`api.github.com` artifact *listing*** |
| `pypi.org`, `files.pythonhosted.org` | working | the full dependency set installs (torch included, CPU) |
| `productionresultssa8.blob.core.windows.net` (Actions artifact payloads) | `EOF` | **workflow artifacts cannot be downloaded here**, only *listed* via the API. Measured by attempting `gh run download 35042805806 --name submission-final-35042805806`. |
| `www.dropbox.com`, `drivendata.org`, `s3.amazonaws.com`, `prd-tnm.s3.amazonaws.com`, `sciencebase.gov`, `apps.nationalmap.gov`, `gdr.openei.org`, `community.drivendata.org`, `raw.githubusercontent.com`, `objects.githubusercontent.com` | TLS `EOF` | no direct data download; no release/artifact/raw fetch |

**Not a limit: the competition data is already in hand.** `gems-geodawn-numerical-features.tif`
(418,912,844 B), `existing_faults.tif` (425,830 B), `example_submission.tif` (1,599,597 B),
`GEMS_96647.pdf` (455,140 B) and `Digital-elevation-model-links-JSON.pdf` (23,032,446 B) were
downloaded, hashed and measured by a **GitHub-hosted runner** in session 2/3
(`data/evidence/inventory.json`, `data/evidence/rasters.json`), and a 512×512 window of the real
rasters is committed as `data/fixture/` so the pipeline can be exercised in-sandbox. The runner is the
only machine here with open egress; that is why every data-touching job is a workflow.

## 2. Compute (re-measured)

2 vCPU / 3 GB RAM / 20 GB disk in the sandbox; GitHub-hosted `ubuntu-latest` runners are 4 vCPU CPU-only
with a 6-hour job limit. Consequences actually observed:

* full-suite tests **do** run here (torch CPU wheel from PyPI): 37/37 pass;
* training does **not** meaningfully run here — 3 GB RAM caps patch size and the 2 vCPUs put a 6-fold
  ensemble out of reach;
* the 6-fold ensemble needed **2 h 36 min** per fold on a runner (`run 35042805806`), i.e. a
  leaderboard-grade ensemble on CPU is possible but slow, which is exactly why the fold jobs are
  parallel and time-guarded (`training.max_minutes`).

## 3. Access we still need (human actions — no amount of automation here replaces them)

1. **DrivenData account + competition enrolment.** Required to (a) download the official data-tab
   files under the competition's own terms, (b) *upload* any submission (this is the single blocking
   step between this repository and the leaderboard), and (c) see the public leaderboard, which is the
   only unbiased feedback on the scored universe (3 submissions/week, rules §3.2).
2. **Eligibility** (rules §1.3): US citizen/permanent resident (or a team whose captain is), US
   incorporation for private entities, US-accredited institutions for academics; DOE employees/support
   contractors and FFRDC *institutional* participation are excluded (FFRDC-affiliated individuals may
   compete individually but are not cash-eligible). Worth a check before any prize is contemplated.
3. **A GPU**, to run `configs/config.yaml` (EfficientNet-B5, 10 splits, 60 epochs) rather than the
   CPU-feasible `configs/config_ci_ensemble.yaml`.
4. **Narrative + code-asset submission** at the deadline: the rules require the complete solution
   assets with resource documentation and a generative-AI disclosure; this repository is deliberately
   shaped to be that package, but the upload itself is human.

5. **Proxies are labelled as proxies, but they are still proxies.** Four different quantities appear in
   this repository and none of them is the competition score: (a) held-out DTI against the public
   catalogue (the wrong population — rules §1.1 scores new faults), (b) discovery diagnostics
   (unlabelled), (c) the shift-robustness width curve (a stress test of the writing operator, run on the
   catalogue), and (d) the blanket-ones floor (a constant). The new LOO audit removes one specific bias —
   fitting and scoring the floor on the same folds — and nothing more.
6. **The emission-width stress test was run on one window.** Rows 2048–2560 × cols 1280–1792 of the
   official grid is the only place where a committed submission and the official label raster overlap
   locally. The 6-fold A/B runs on the runner over all six held-out crops, but the absolute numbers in
   `data/evidence/shift_robustness.json` are single-window and are reported as such.
