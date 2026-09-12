# GEMSDOE — Geologic Enhanced Mapping System Prize Challenge
### Top-Leaderboard Solution Framework

**Competition:** [GEMS Prize Challenge on DrivenData](https://www.drivendata.org/competitions/306/competition-doe-gems/)  
**Problem Description:** https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/  
**About / Resources:** https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/  
**Data Tab (requires login):** https://www.drivendata.org/competitions/306/competition-doe-gems/data/  
**Official Rules (HeroX):** https://www.herox.com/GEMSPrize/resource/2274 → PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf  
**Reference Solution:** https://github.com/drivendataorg/gems-prize-reference-solution  
**Prize:** $300,000 total ($50k initial, $250k final) | End Date: Dec 3, 2026 11:59pm UTC  
**Sponsor:** DOE Office of Geothermal + National Lab of the Rockies

---

## 1. Problem Summary (verified, no hallucinations)

**Task:** Develop models that predict presence of geological faults (fractures indicative of geothermal resources) from geophysical data in the **GeoDAWN region** (Geoscience Data Acquisition for Western Nevada).

- **GeoDAWN** = high-resolution airborne magnetic + radiometric surveys in NW Nevada & E California, covering Walker Lane and western Great Basin. Source: USGS Data Release [https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7), DOI [https://doi.org/10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ), and overview [https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)

- **Labels:** USGS Quaternary Fault and Fold Database + INGENIOUS project. Known faults are incomplete; challenge provides new expert-labeled hidden faults for test. Final scoring uses expanded label set after expert review of all submissions.

- **Provided Features (training_features.tif, UTM 11N EPSG:32611, 100m resolution):**
  - Surface conductivity & depth to conductive base
  - Detrended elevation & slope of detrended elevation
  - Dilatation rate, shear strain rate, second invariant of strain rate tensor
  - Isostatic gravity anomaly & slope of isostatic gravity anomaly
  - Magnetics: reduced-to-pole magnetic anomaly, total magnetic intensity, vertical & horizontal slope of TMI, top-of-crustal magnetic source depth estimate
  - Density of earthquakes
  - Plus CSV `1m_DEM_links.csv` for high-res DEM download

- **Metric:** Distance-weighted Tversky index with triangular kernel R=300m (3 pixels), alpha=0.2 (FP penalty low), beta=0.8 (FN penalty high). Rewards high recall, tolerates small misalignment.

  Formulas (from problem page):
  - k(d) = max(1 - d/R, 0)
  - TP_w = Σ_{g∈G} max_{x:d(x,g)≤R} p(x) k(d(x,g))
  - FP_w = Σ_{x:p(x)>0} p(x) [1 - max_{g∈G} k(d(x,g))]
  - FN_w = Σ_{g∈G} [1 - max_{x:d(x,g)≤R} p(x) k(d(x,g))]
  - DTI = TP_w / (TP_w + α FP_w + β FN_w + ε)

- **Submission:** Single-layer GeoTIFF float32 in [0,1], same CRS (EPSG:32611), same resolution (100m), same bounds as training, NaN outside.

- **Competition Structure:** Same submission scored twice:
  - Initial Round: private pre-labeled hidden faults (public leaderboard = public subset)
  - Expert review uses all submissions to find candidate new faults → expanded label set
  - Final Round: rescore against expanded set → true discoveries rewarded.

---

## 2. Limitations & Required Access (as requested)

### Current Limitations in this Sandbox
- **No DrivenData authentication:** Cannot download `training_features.tif`, `labels.tif`, `sample_submission.tif`, `1m_DEM_links.csv` from https://www.drivendata.org/competitions/306/competition-doe-gems/data/ without login. Code is built to work once user places data in `data/`.
- **No GPU / limited CPU:** Training large segmentation models (U-Net, SegFormer) ideally needs GPU (CUDA or Apple MPS). Environment here is CPU-only Python 3.11.2, no torch installed initially.
- **No large external data pre-downloaded:** GeoDAWN grids are GB-scale, 1m DEM tiles are many GB. We provide download scripts but cannot bulk-download here.
- **Private test labels unavailable:** Expected; we must use cross-validation and visual inspection.

### What We Need for Full Competitive Run
1. **DrivenData account + competition enrollment** to download official data.
2. **Compute:** GPU machine (e.g., 1x A100, 24GB+ VRAM) or multi-GPU for ensemble training. Reference solution recommends CUDA 12.6 or 13.0.
3. **Storage:** ~50GB for GeoDAWN + DEM + INGENIOUS + processed patches.
4. **Optional:** Access to USGS AWS Open Data for 3DEP lidar: https://registry.opendata.aws/usgs-lidar/
5. **Time:** ~5-10h training for full ensemble (MC=10, 50 epochs).

### What Is Allowed for External Data
Per rules: any data with license permitting use in challenge and sharing with sponsor for evaluation. All our listed external sources are **public domain (U.S. Government)** or CC-licensed.

---

## 3. Verified External Data Catalog (Auditable, No Hallucinations)

All entries have official verified links for manual review. See also `docs/data_catalog.csv` and `docs/data_catalog.json`.

| # | Dataset Name | Official Source | Verified Link(s) | License | Relevance | How Used |
|---|--------------|-----------------|------------------|---------|-----------|----------|
| 1 | **GeoDAWN Airborne Magnetic & Radiometric Surveys** | USGS | ScienceBase: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 <br> DOI: https://doi.org/10.5066/P93LGLVQ <br> USGS overview: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and | Public Domain (USGS) | Primary geophysics in challenge | Provides high-res mag & radiometric; we use to augment training_features |
| 2 | **USGS Quaternary Fault & Fold Database** | USGS Earthquake Hazards | Interactive: https://www.usgs.gov/programs/earthquake-hazards/faults <br> ScienceBase: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 <br> DOI: https://doi.org/10.5066/P9BCVRCK <br> ArcGIS: https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer | Public Domain | Ground truth labels | Training labels source; we verify alignment, use for additional supervision |
| 3 | **USGS 3DEP 1m DEM / Lidar** | USGS National Geospatial Program | About: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services <br> Downloader: https://apps.nationalmap.gov/downloader/ <br> LidarExplorer: https://apps.nationalmap.gov/lidar-explorer/ <br> AWS: https://registry.opendata.aws/usgs-lidar/ | Public Domain ("free of charge and without use restrictions") | Elevation detail | Official source for `1m_DEM_links.csv`; we compute slope, curvature, TRI, TPI, hillshade |
| 4 | **INGENIOUS Great Basin Regional Dataset Compilation** | GBCGE / DOE GDR | GDR: https://gdr.openei.org/submissions/1391 <br> DOI: https://doi.org/10.15121/1881483 <br> OSTI: https://www.osti.gov/biblio/1881483 <br> Project: https://gbcge.org/current-projects/ingenious/ | **CC BY 4.0** (verified on GDR page) | Features in challenge (conductivity, strain, gravity, etc.); also origin of training labels per rules PDF §3.3 | Source for conductivity, strain rate, gravity, earthquake density layers; sub-datasets have own USGS DOIs (P9TWT2LU MT conductance, P9MQRCBY detrended elevation, P9Z6SA1Z gravity/magnetics, P9BZPVUC heat flow, P9YL58W6 slip/dilation tendency) |
| 5 | **GBCGE Subsurface Database Explorer** | GBCGE | DOI: https://doi.org/10.15121/1987556 <br> OSTI: https://www.osti.gov/dataexplorer/biblio/dataset/1987556 | Public | Geothermal wells, springs, temps | External features: temp probes, well chemistry |
| 6 | **Geothermal Favorability – INGENIOUS Features** | USGS + INGENIOUS | ScienceBase: https://www.sciencebase.gov/catalog/item/66e88690d34e0606a9db9b43 | Public | ML favorability | 16 input features curated for publication |
| 7 | **USGS EarthMRI** | USGS Mineral Resources | Portal: https://mrdata.usgs.gov/earthmri/ <br> Overview FS: https://pubs.usgs.gov/publication/fs20203055 | Public Domain | Critical mineral context | Understanding acquisition areas |
| 8 | **INGENIOUS Papers & Methods** | Academia | Mattéo et al 2021: https://doi.org/10.1029/2020JB021269 <br> Hermant et al 2025: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf | Open Access | Method inspiration | Automatic fault mapping with deep learning |

**No hallucinations:** Every link above was verified via web_search and fetch_page during development. See `docs/references.md` for fetch logs.

**Irregularities flagged:**
- Competition data tab requires login – cannot verify exact file sizes/names without account. Reference solution mentions `numeric_features.tif` vs `training_features.tif` naming drift – we handle both.
- 1m DEM links CSV likely points to The National Map – we provide fallback via LidarExplorer.
- Some INGENIOUS layers (e.g., detrended elevation) derivation not fully documented – we re-derive from 3DEP.

---

## 4. Solution Strategy to Place Top of Leaderboard

### Why Reference Solution is Insufficient
- Single U-Net, 5 MC splits, 5 epochs, basic normalization, no external DEM, no post-processing, no TTA, no ensemble blending, no distance-weighted loss approximation.

### Our Improvements (implemented in `src/`)

#### A. Data Engineering
- **Robust normalization:** per-channel nan-aware clipping (2-98 percentile) + standardization, not just [0,1] min-max.
- **External DEM pipeline:** Download 1m DEM tiles, resample to 100m, compute:
  - Slope, aspect, curvature (profile & plan), Topographic Position Index (TPI), Terrain Ruggedness Index (TRI), hillshade, detrended elevation via Gaussian filter.
  - These augment provided features; official source: USGS 3DEP.
- **Patching:** Overlapping patches (train_step=32) vs non-overlap test, filter patches with fault presence + valid elevation, plus hard-negative mining.
- **Augmentation:** RandomResizedCrop (0.5-1.0), flips, rotations, elastic, Gaussian noise, brightness/contrast per channel.

#### B. Model Architecture
- **Encoders:** EfficientNet-B3/B5, ResNet-50, MIT-B2 (SegFormer) pretrained on ImageNet.
- **Decoders:** UNet++, DeepLabV3+, SegFormer, UPerNet.
- **Loss:** TverskyLoss (α=0.2, β=0.8) + Focal Tversky + Dice + BCE combo. Weighted to match metric.
- **Training:** 
  - MC cross-validation: 10 splits, 70/30, stratified by fault density.
  - Mixed precision, AdamW, OneCycleLR (init 1e-4), early stopping on distance-weighted Tversky.
  - Batch 32, patch 256 (larger context than 128).
- **Inference:** Sliding window with 50% overlap + Gaussian blending, Test-Time Augmentation (8 flips/rotations), ensemble averaging across MC splits and architectures.

#### C. Post-Processing for Final Round (Discover New Faults)
- **Thresholding:** Low threshold (0.1-0.2) to maximize recall (β=0.8).
- **Morphology:** Skeletonization, dilation, closing to connect fault segments.
- **Line enhancement:** Frangi filter, Hough transform for linear continuity.
- **False positive control:** Mask with slope & curvature thresholds (faults often correlate with topographic lineaments).

#### D. Metric Implementation
- Exact reproduction of distance-weighted Tversky with triangular kernel R=300m in `src/metrics.py`, using scipy distance transforms for efficiency.
- Used for validation and early stopping.

#### E. External Data Fusion (allowed)
- All external data from table above, public domain.
- We never use private test labels.

### Expected Gains (clearly labeled estimates — not verified facts)
- Reference baseline: untested in this environment; expected lower than a tuned ensemble (single U-Net, 5 epochs, no TTA/ensembling — per cloned notebook).
- Our ensemble + external DEM + TTA: **target** 0.65+ DTI in local MC cross-validation. There is no public history for this new competition, so no leaderboard correlation is claimed; the public LB will be the first real signal (3 submissions/week budget).

---

## 5. Repo Structure

```
GEMSDOE/
├── README.md (this file)
├── docs/ (GitHub Pages site)
│   ├── index.html (clean UI)
│   ├── style.css
│   ├── data_catalog.csv
│   ├── data_catalog.json
│   ├── references.md
│   └── assets/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── inference.py
│   ├── postprocess.py
│   └── external_data.py
├── configs/
│   └── config.yaml
├── scripts/
│   ├── download_external.sh
│   └── prepare_data.py
├── requirements.txt
├── environment.yml
└── .github/workflows/pages.yml
```

---

## 6. Quickstart (once you have DrivenData data)

```bash
# 1. Clone
git clone https://github.com/buffedlizard55-lab/GEMSDOE.git
cd GEMSDOE

# 2. Env (conda or pip)
conda env create -f environment.yml
conda activate gemsdoe
# or
pip install -r requirements.txt

# 3. Place competition data
mkdir -p data/
# Download from https://www.drivendata.org/competitions/306/competition-doe-gems/data/
# Expected files:
# - training_features.tif (or numeric_features.tif)
# - labels.tif (or faults.tif)
# - sample_submission.tif
# - 1m_DEM_links.csv

# 4. Optional external data
bash scripts/download_external.sh  # downloads INGENIOUS sample, QFaults shapefile

# 5. Train
python -m src.train --config configs/config.yaml

# 6. Inference -> submission.tif
python -m src.inference --config configs/config.yaml --model-dir outputs/best --out submission.tif

# 7. Evaluate (if you have val labels)
python -m src.metrics --pred submission.tif --true data/labels.tif
```

---

## 7. GitHub Pages

We provide a clean, user-friendly site in `docs/`:

- Deploy: Settings → Pages → Source: GitHub Actions (workflow in `.github/workflows/pages.yml`)
- Local preview: `cd docs && python -m http.server 8000`
- Site includes:
  - Problem overview
  - Metric visualization
  - Auditable data catalog with official links
  - Model architecture diagram
  - Limitations & needed access
  - References

URL after deploy: `https://buffedlizard55-lab.github.io/GEMSDOE/`

---

## 8. Verification & No Hallucinations Statement

- All external links were fetched via `fetch_page` or `web_search` during development.
- Data catalog entries have DOIs or USGS official pages.
- Code for metric matches problem description formulas exactly.
- No synthetic fault data invented.
- Flagged irregularities in section 3.

---

## 9. License & Citation

- Code: MIT (as reference solution)
- Data: Respective licenses (USGS Public Domain, INGENIOUS CC)
- If using this repo, cite:
  - Glen & Earney 2024 GeoDAWN DOI 10.5066/P93LGLVQ
  - USGS QFaults DOI 10.5066/P9BCVRCK
  - INGENIOUS DOI 10.15121/1881483
  - Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/

---

## 10. Contact / Next Steps

- TODO for owner: Add DrivenData API token to download data automatically (if allowed).
- TODO: Run full training on GPU and submit to leaderboard.
- TODO: Expert review contribution – our predictions aim to maximize true new fault discovery for Final Round.

Built with ❤️ for geothermal discovery.
