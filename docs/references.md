# Verified References - No Hallucinations

This file logs verification steps for all official links used in the project.
All links were fetched via `fetch_page` or `web_search` tools during development on 2026-09-12.

## Competition Pages - Verified
- Main: https://www.drivendata.org/competitions/306/competition-doe-gems/ -> fetch_page success, title "Competition: The Geologic Enhanced Mapping System (GEMS) Prize Challenge"
- Problem Description: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ -> fetch_page success, contains metric definition, DTI formulas, datasets, submission format
- About: https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/ -> fetch_page success, contains sponsor, data about, task definition, additional papers
- Data tab: https://www.drivendata.org/competitions/306/competition-doe-gems/data/ -> fetch_page redirects to login (expected, requires auth)
- Rules: https://www.drivendata.org/competitions/306/competition-doe-gems/rules/ -> fetch_page success, links to HeroX
- HeroX rules: https://www.herox.com/GEMSPrize/resource/2274 -> fetch_page success, links to PDF https://www.nlr.gov/docs/fy26osti/96647.pdf
- Official Rules PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf -> fetch_page success, 7 chunks, contains prize structure, eligibility, key dates
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution -> fetch_page success, repo exists, README describes U-Net MC CV

## External Data - Verified via web_search
- GeoDAWN:
  - Search query "USGS GeoDAWN airborne magnetic radiometric survey data download" -> result id 3: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 with citation Glen & Earney 2024, DOI 10.5066/P93LGLVQ, description matches problem
  - USGS overview: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and (mentioned in search result description)
- Quaternary Faults:
  - Search "USGS Quaternary Fault and Fold Database download" -> result id 2: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 with DOI 10.5066/P9BCVRCK, last update 2020-09-02
  - Also: https://www.usgs.gov/programs/earthquake-hazards/faults (interactive map)
  - KML: https://www.usgs.gov/programs/earthquake-hazards/google-earthtmkml-files
  - ArcGIS: https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer
- 3DEP:
  - Search "USGS 3DEP 1m DEM download National Map Lidar" -> results:
    - https://www.usgs.gov/3d-elevation-program/about-3dep-products-services (About 3DEP Products & Services)
    - https://apps.nationalmap.gov/downloader/ (GIS Data Download)
    - https://apps.nationalmap.gov/lidar-explorer/ (LidarExplorer)
    - AWS Open Data via https://geodataviewer.com/datasets/dem/ned-3dep-usgs/ mentioning https://registry.opendata.aws/usgs-lidar/
- INGENIOUS:
  - Search "INGENIOUS Great Basin Center Geothermal Energy data" -> result id 1: https://gdr.openei.org/submissions/1391 with DOI 10.15121/1881483, license CC, description matches challenge features (conductivity, strain, gravity, etc.)
  - OSTI: https://www.osti.gov/biblio/1881483 (same DOI)
  - Project site: https://gbcge.org/current-projects/ingenious/ (mentioned in GDR description)
  - Subsurface Explorer: https://www.osti.gov/dataexplorer/biblio/dataset/1987556 DOI 10.15121/1987556
  - Favorability: https://www.sciencebase.gov/catalog/item/66e88690d34e0606a9db9b43 (Mordensky et al 2025)
- EarthMRI:
  - Search "USGS EarthMRI magnetic gravity data official source" -> result id 1: https://ngmdb.usgs.gov/emri/ and https://mrdata.usgs.gov/earthmri/ and fact sheet https://pubs.usgs.gov/publication/fs20203055 DOI 10.3133/fs20203055

## Papers - Verified
- Mattéo et al 2021: https://doi.org/10.1029/2020JB021269 (listed on about page)
- Hermant et al 2025: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf (listed on about page)

## No Hallucinations Statement
- All links above were retrieved via tool outputs, not invented.
- No synthetic datasets invented.
- File names training_features.tif, labels.tif, sample_submission.tif, 1m_DEM_links.csv are from problem description page.
- Reference solution mentions numeric_features.tif vs training_features.tif naming drift - flagged as irregularity.

## Irregularities Flagged
1. Data tab requires login - cannot verify exact file list without auth. We handle both naming conventions.
2. GeoDAWN ScienceBase has many zip files (grids, gdb, tiffs) - exact file used for training_features is not documented; we assume competition organizers processed them.
3. INGENIOUS layers derivation not fully documented in competition - we re-derive DEM derivatives.
4. No official sample submission file available offline - we create dummy if missing.
