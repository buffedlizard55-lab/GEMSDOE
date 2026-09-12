# data/ — Competition Data: ACQUISITION STATUS (updated 2026-09-12)

| File | Status | Where |
|---|---|---|
| `1m_DEM_links` (JSON in PDF) | ✅ **PARTIALLY CAPTURED** (chunk 1/13; Dropbox `st` signature expired mid-fetch) — 35 unique DEM tile URLs extracted, 6 S3-verified (1,316 MB of tiles confirmed with exact sizes/ETags) | `data/dem_links.json`, evidence in `data/evidence/` |
| `GEMS_96647.pdf` (rules mirror) | ✅ **IDENTITY VERIFIED** — fetched via platform; content matches canonical `https://www.nlr.gov/docs/fy26osti/96647.pdf` (not duplicated here; canonical already verified) | fetch log: `data/evidence/dropbox_fetch_log.md` |
| `gems-geodawn-numerical-features.tif` | ❌ **NOT DOWNLOADED** — sandbox egress allowlist blocks Dropbox/S3; binary too large for text fetch | run `bash scripts/download_competition_data.sh` on any unrestricted machine |
| `existing_faults.tif` (labels) | ❌ NOT DOWNLOADED — same | same |
| `example_submission.tif` | ❌ NOT DOWNLOADED — same | same |

DEM tile URLs point at the official USGS bucket `prd-tnm.s3.amazonaws.com` (projects `CA_SierraNevada_B22`, `NV_WestCentral_EarthMRI_2020_D20`, `NV_Humboldt_2021_D21`). The full authoritative tile list can be enumerated without the competition PDF: `python scripts/download_dem_tiles.py --complete-listing`.

Irregularities found inside the competition DEM-links JSON (documented, S3-proven): duplicate rows (up to 5×), path/filename project mismatches (canonical URL uses the FILENAME project — the mismatched path variant returns key_count=0), garbled host variants in the PDF print (prdtnm / prd.tnm / prd- tnm).

---

# data/ — Competition Data Placement Guide

All competition data requires **DriventData login + enrollment**: https://www.drivendata.org/competitions/306/competition-doe-gems/data/

This sandbox cannot download them (no DrivenData account; TLS to Dropbox/DrivenData blocked — see `LIMITATIONS.md`). Download manually and place files here.

## Expected files and naming

⚠️ **Naming drift flagged (irregularity):** the same content appears under three different names across official sources. `src/dataset.py` and `scripts/prepare_data.py` accept **all** variants.

| Content | Problem page name | Reference solution name | Dropbox mirror name (from data tab) |
|---|---|---|---|
| Input features (multiband GeoTIFF, EPSG:32611, 100 m) | `training_features.tif` | `numeric_features.tif` | `gems-geodawn-numerical-features.tif` |
| Training labels (fault raster) | `labels.tif` | `labels.tif` | `existing_faults.tif` |
| Submission template | `sample_submission.tif` | — (derives from labels) | `example_submission.tif` |
| 1 m DEM link list | `1m_DEM_links.csv` | — | `Digital-elevation-model-links-JSON.pdf` |
| Official rules PDF | hosted at https://www.nlr.gov/docs/fy26osti/96647.pdf | — | `GEMS_96647.pdf` |

## Dropbox mirrors (competition-provided, from data tab)

> Provenance: supplied by the competition/user; this sandbox could not fetch Dropbox (TLS-blocked), so contents are not independently verified — the identical rules PDF was verified at the NLR URL above.

- Rules PDF: https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=1
- Example submission: https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=1
- Existing faults (labels): https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=1
- GeoDAWN numerical features: https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=1
- DEM links (JSON PDF): https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=1

(Swap `dl=0` → `dl=1` for direct download.)

## After download — sanity checks (no hallucination policy)

```bash
python scripts/prepare_data.py        # validates presence, CRS, resolution, bounds
python scripts/validate_submission.py # run after generating a submission
```

Expected (per official problem description):
- Features/labels: EPSG:32611, 100 m resolution, same bounds.
- Labels: single band, positive pixels = fault.
- Submission template: single band float32, values in [0,1].

## External data (all public domain / CC BY 4.0 — allowed by rules)

See `docs/data_catalog.csv` (one row per link, with verification status) and `scripts/download_external.sh`.

Primary external sources (official):
- GeoDAWN raw surveys (CC0): https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
- INGENIOUS regional compilation (CC BY 4.0): https://gdr.openei.org/submissions/1391
- USGS Qfaults (public domain): https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
- 1 m DEM: https://apps.nationalmap.gov/downloader/ (or links in `1m_DEM_links.csv` once available)
