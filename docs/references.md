# Verification Log — All Links Checked, No Hallucinations

**Verification date:** 2026-09-12 (autonomous, tool-driven)
**Methods:** `fetch_page` (platform-side full-text retrieval), `web_search` (result titles + content cross-checked), `git clone` (reference solution).
**Environment note:** this sandbox has **no direct TLS to external hosts** (curl/urllib fail with SSL EOF to every non-git host), so all HTTP verification runs through the platform fetch/search tools. Files could not be bulk-downloaded. See `LIMITATIONS.md`.

Evidence standard used: a link is marked **VERIFIED** only if (a) fetched full-text, (b) git-cloned, (c) hit by search results with matching title/content, or (d) explicitly marked LINKED-FROM-OFFICIAL-PAGE / USER-PROVIDED. Everything else was **removed** and logged below.

---

## 1. Competition chain — all fetched in full

| Step | URL | Result |
|---|---|---|
| Main page | https://www.drivendata.org/competitions/306/competition-doe-gems/ | ✅ fetched — end date **Dec 3, 2026 11:59 p.m. UTC**, $300k pool ($50k initial top-5 × $10k; $250k final $100k/$70k/$40k/$25k/$15k), eligibility (US citizen/PR; team captain US citizen/PR), sponsor DOE Office of Geothermal + **National Lab of the Rockies (NLR)**, contact **gemsprize@nlr.gov** |
| Problem description | https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ | ✅ fetched (2 chunks) — task, competition structure, datasets, full metric math (triangular kernel R=300 m, α=0.2, β=0.8, TP_w/FP_w/FN_w formulas + worked example TI=0.60), submission format (float32 [0,1], EPSG:32611, 100 m, same bounds, NaN outside) |
| About | https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/ | ✅ fetched — sponsor, GeoDAWN background, task background, 2 recommended papers |
| Data tab | https://www.drivendata.org/competitions/306/competition-doe-gems/data/ | ⚠️ **redirects to login** (matches user-reported behavior) — file list not enumerable without enrolled account |
| Rules page | https://www.drivendata.org/competitions/306/competition-doe-gems/rules/ | ✅ fetched — single link to HeroX |
| HeroX resource 2274 | https://www.herox.com/GEMSPrize/resource/2274 | ✅ fetched — page "GEMS Prize Official Rules", links to NLR PDF |
| Official rules PDF | https://www.nlr.gov/docs/fy26osti/96647.pdf | ✅ fetched (7 chunks) — "Geologic Enhanced Mapping System (GEMS) Prize Official Rules", **September 2026**, governed by 15 U.S.C. § 3719. Key sections verified: 1.1 phases/prizes, 1.3 eligibility (incl. FFRDC researchers may compete individually but honorable-mention only), 1.4 prize goals, 3.2 process (single GeoTIFF; **three submissions per week**; **generative-AI disclosure required in narrative**), 3.3 training labels "obtained from the INGENIOUS project's Great Basin Regional Dataset Compilation" (footnote 4 = Ayling et al. 2022, doi:10.15121/1881483; footnote 3 = Glen & Earney 2024 GeoDAWN doi:10.5066/P93LGLVQ), 3.4 feedback (3/week, exactly one final), 3.5 one submission for both rounds + complete code assets/docs per "DrivenData Winning Model Documentation Template", 3.6.1 public/private split, 3.6.2 choose single submission without private knowledge, 3.6.5 winner notification ~60 days, A.1 5:00 p.m. ET deadline, A.2 ACH + W-9 within 30 days |
| Reference solution | https://github.com/drivendataorg/gems-prize-reference-solution | ✅ **git cloned** — author John Lipor (PSU); notebook uses `data/numeric_features.tif` + `data/labels.tif`; smp U-Net, MC=5, patch 128, test_proportion 0.5, batch 32, epochs 5, lr 1e-4, Tversky α0.2/β0.8; uv extras cu126/cu130/cpu |

## 2. Official data sources — fetched in full

- **GeoDAWN ScienceBase** https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 — ✅ fetched (2 chunks). Citation "Glen, J.M.G., and Earney, T.E., 2024", DOI 10.5066/P93LGLVQ, publication 2024-03-01. **Full file list with exact sizes captured** (e.g. 22103_area1_grids.zip 42.43 MB; 22103_area2_grids.zip 227.39 MB; 22103_mag_a1_gdb.zip 350.44 MB; 22103_spec_a1_gdb.zip 378.61 MB; 22103_area1_tiffs.zip 43.57 MB; 22103_area2_tiffs.zip 230.54 MB; 22103_mag_a2_csv.zip 3.74 GB). Survey facts: 149,030 line-km; 51,857 km²; blocks Winnemucca/Fallon/Hawthorne/Tonopah; EDCON-PRJ.
- **GeoDAWN USGS overview** https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and — ✅ fetched. **Rights: CC0 1.0 Universal.**
- **Qfaults ScienceBase** https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 — ✅ fetched. Citation "U.S. Geological Survey, 2020, Quaternary Fault and Fold Database for the Nation", DOI 10.5066/P9BCVRCK, last update 2020-09-02; files qfaults.kmz 18.97 MB, Qfaults_GIS.zip 30.87 MB; related links: faults page + ArcGIS MapServer + fact sheet FS2004-3033.
- **3DEP about** https://www.usgs.gov/3d-elevation-program/about-3dep-products-services — ✅ fetched. "All 3DEP products are available, free of charge and without use restrictions."
- **AWS 3DEP lidar** https://registry.opendata.aws/usgs-lidar/ — ✅ fetched. "US Government Public Domain"; s3://usgs-lidar-public (free EPT) + s3://usgs-lidar (requester-pays LAZ).
- **The National Map Downloader** https://apps.nationalmap.gov/downloader/ — ✅ via USGS pages (media screenshot page names the URL).
- **LidarExplorer** https://apps.nationalmap.gov/lidar-explorer/ — ✅ via 3DEP page quick links (usgs.gov/NationalMap/LidarExplorer) + corroboration.
- **INGENIOUS GDR 1391** https://gdr.openei.org/submissions/1391 — ✅ fetched. DOI 10.15121/1881483; **license CC BY 4.0**; 116.98 MB / 9 files; all 16 resources captured, including linked USGS sub-datasets with their own DOIs: **P9TWT2LU** (MT conductance), **P9MQRCBY** (elevation trend & detrended elevation), **P9Z6SA1Z** (gravity & magnetics), **P9BZPVUC** (heat flow), **P9YL58W6** (Q faulting slip/dilation tendency), and Thermal Conductivity → GDR 1390.
- **INGENIOUS OSTI** https://www.osti.gov/biblio/1881483 — ✅ fetched. 23-author list matches rules PDF footnote 4 exactly; 30 June 2022.
- **GBCGE Subsurface Explorer** https://www.osti.gov/dataexplorer/biblio/dataset/1987556 — ✅ fetched. Mlawsky & Ayling, 2020-03-15, DOI 10.15121/1987556.
- **Favorability** https://www.sciencebase.gov/catalog/item/66e88690d34e0606a9db9b43 — ✅ fetched. Mordensky & DeAngelo 2025, DOI **10.5066/P14EET2C**, 16 input features, zip 1.22 GB, companion paper DOI 10.1016/j.geothermics.2025.103450 (Mordensky, Burns, **Lipor**, DeAngelo — Geothermics 133, 103450).
- **EarthMRI portal** https://mrdata.usgs.gov/earthmri/ — ✅ fetched.
- **EarthMRI fact sheet** https://pubs.usgs.gov/publication/fs20203055 — ✅ search-verified: Day, W., 2020, FS 2020-3055, 4 p.
- **Isostatic gravity** https://mrdata.usgs.gov/metadata/usgraviso.html + https://mrdata.usgs.gov/gravity/isostatic/ — ✅ search-verified (Kucks 1999; 2.67 g/cc Bouguer; use constraints none).
- **USGS FAQ (fault map GIS/KML)** https://www.usgs.gov/faqs/where-can-i-find-a-fault-map-united-states-one-available-gis-format — ✅ search-verified (KML + shapefiles confirmed).

## 3. Literature — search/fetch-verified with full citations

- ✅ Mattéo et al 2021 — doi:10.1029/2020JB021269 (listed on official About page).
- ✅ Hermant et al 2025 — Stanford SGW PDF (listed on official About page).
- ✅ Lei et al 2025 TransVNet — Front. Earth Sci. **13:1635344**, published **08 Aug 2025**. *(Corrected: previously mis-dated "Dec 27 2025".)*
- ✅ Dalton et al 2025 Geo-SegNet — Computers & Geosciences (S2949673X25000026). *(Corrected: it is geomaterial micro-CT pore segmentation; earlier description over-claimed fault relevance.)*
- ✅ Zhong et al 2024 Frangi palaeochannel — GJI **236(3):1526-1544**, doi:10.1093/gji/ggad491, online 22 Dec 2023.
- ✅ Guo et al 2025 multi-source fusion fault identification — Sci Rep **15:6643**, 24 Feb 2025, PMC11850705, authors Guo/Yang/Peng/Zhu/He; **erratum** Sci Rep 15:14821 (28 Apr 2025) noted.
- ✅ Chukwu et al 2024 Euler+DBSCAN — **Exploration Geophysics 55(3):223-245**, doi:10.1080/08123985.2023.2299475, online 03 May 2024. *(Corrected from "Tandfonline 2023".)*
- ✅ Wang & AlRegib 2014 — IEEE 6854024, ICASSP, pp. 2372-2376, doi:10.1109/ICASSP.2014.6854024; free PDF mirror at dihana.cps.unizar.es. *(gatech wpmucdn mirror removed — unverified.)*
- ✅ Hassan & Goussev 2019 — GeoConvention GC2019_313 PDF (authors confirmed from PDF hit).
- ✅ Bai et al 2024 — Remote Sensing 16(7):1115, doi:10.3390/rs16071115.
- ✅ Rafiq et al 2025 — MethodsX 14:103189, PMC11834046 (CET + FFD protocol).
- ✅ Guo et al 2021 — Computers & Geosciences 149:104701 (Noddy magnetic inversion).
- ✅ Jessell et al 2022 — ESSD 14:381 (Noddyverse, 1M models).
- ✅ Feng, Hu, Zhao, Ai 2025 — Huanggang City two-stage ML/DL; metrics F1 90.91%/GDA 2.82% and RF-MUnet F1 92.47%/GDA 1.94% confirmed verbatim.
- ✅ Florsch et al 2022 — Computers & Geosciences 169:105227 (YOLO+DenseNet, Grad-CAM, t-SNE).
- ✅ Sun, Ran, Xue et al 2026 — Earth Sci Inform **19:173**, GWSU-Net, 19 maps, 32,582 patches 128×128, F1 0.8634/0.8504.
- ✅ GSRM v2.1 — https://gsrm2.unavco.org/ (cite Kreemer, Blewitt & Klein 2014, doi:10.1002/2014GC005407).
- ✅ Hammond, Kreemer & Blewitt 2024 — JGR Solid Earth 129, e2023JB028044.
- ✅ Faulds & Henry 2008 — AGS Digest 22, p. 437-470 (PDF fetched; ~20% of Pacific–North America motion; ~10 mm/yr GPS strain across WL-ECSZ).

## 4. Corrections & removals applied this review (flag → fix)

| Removed / fixed | Reason | Replacement |
|---|---|---|
| ❌ `pubs.usgs.gov/pp/1720/downloads/pdf/p1720D.pdf` | Exact PDF could not be verified to exist | RTP attributed to Baranov & Naudy 1964 (as cited in verified USGS OFR text) |
| ❌ `agupubs.../10.1002/2014JB011145` described as "Steady contemporary deformation" | Search shows different title for that DOI — mislabeling risk | Removed; kept verified 10.1029/2023JB028044 |
| ❌ `geodataviewer.com/...` | Third-party, not official | Official 3DEP links only |
| ❌ `ngmdb.usgs.gov/emri/` | Unverified | Official portal https://mrdata.usgs.gov/earthmri/ |
| ❌ `usgs.gov/programs/earthquake-hazards/google-earthtmkml-files` | Exact URL unverified | USGS FAQ page + verified `qfaults.kmz` on ScienceBase |
| ❌ gatech wpmucdn Wang PDF mirror | Unverified | IEEE DOI + unizar PDF mirror (both verified) |
| ✏️ TransVNet date | "Dec 27 2025" → published 08 Aug 2025 | — |
| ✏️ Euler/DBSCAN citation | "Tandfonline 2023" → Exploration Geophysics 55(3):223-245, 2024 | — |
| ✏️ Geo-SegNet description | Scope corrected to geomaterial segmentation | — |
| ✏️ Licenses | GeoDAWN = CC0 1.0; INGENIOUS = CC BY 4.0 | — |
| ✏️ "3 submissions/week in section 3.4" | Confirmed 3.4 (Feedback) and also stated in 3.2 (Process Overview) | — |
| ✏️ Sci Rep paper | Added authors + erratum | — |

## 5. Remaining flags (cannot be resolved from this sandbox)

1. **Competition data files** — need DrivenData login (C4). Mirror links (D5–D9) are user-provided from the data tab; sandbox TLS policy blocks Dropbox, so their contents were not independently fetched. Filenames differ across three official sources (problem page vs reference solution vs mirrors) — `src/dataset.py` and `scripts/prepare_data.py` handle all variants.
2. **Band-by-band meaning of `gems-geodawn-numerical-features.tif`** — the exact band list/order will be read from the GeoTIFF's band descriptions once downloaded (`src/dataset.py` prints them); the problem page's prose list is the authoritative expectation.
3. **Exact GeoDAWN processing provenance for the detrended-elevation band** — competition organizers' derivation not documented publicly; INGENIOUS DOI P9MQRCBY ("Elevation Trend and Detrended Elevation") is the closest official source.
