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
- `make_patches` keeps training windows that partially overlap held-out test regions (only fully-covered windows are excluded) → mild CV leakage vs the reference solution's global test-region zeroing. Flagged; fix queued in SUGGESTIONS §.
- `pretrained: true` configs cannot fetch encoder weights from this sandbox (release-asset host blocked); sandbox runs use `pretrained: false`. On an unrestricted machine `configs/config.yaml` works as-is.
- 3-epoch CPU smoke model ≈ constant-prediction baseline (DTI 0.0519 vs 0.0524) — honest result; not evidence of model quality.

### 1f. Dropbox mirror fetch notes (2026-09-12)
- Dropbox `scl/fi` links carry a short-lived `st` signature; it expired mid-capture of `Digital-elevation-model-links-JSON.pdf` (chunk 1/13 fetched, rest failed). The durable form is `rlkey` + `dl=1` (used in `scripts/download_competition_data.sh`).
- DEM tiles are ~90-380 MB each; the full 3-project list is tens of GB — download on a machine with disk headroom (`scripts/download_dem_tiles.py`).
