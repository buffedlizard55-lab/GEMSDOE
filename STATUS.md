# Project status — 2026-09-16 (session 4, extended by session 5 below)

Supersedes the previous `STATUS.md` (session 3, 2026-09-15). Session 3's own claims are quoted where
this session falsified them, because the way they were falsified is the most useful thing in this file:
**a green workflow was treated as evidence.**

---

## 1. What this session found: the ensemble run had produced no submission

Session 3 reported: *"On success the blend job commits `data/evidence/runs/35042805806/` (blend_report.json,
validation.log, submission.tif) back to this branch; the artifact `submission-final-*` is the file to
upload to DrivenData."* It also reported *"full suite 23/23 green"*.

Audited against the committed bytes (not the logs):

| artifact | claimed | measured |
|---|---|---|
| `data/evidence/runs/35042805806/submission.tif` | the submission | **110 bytes** — a GDAL stub, not a raster |
| `data/evidence/runs/35042805806/validation.log` | a validation report | **an argparse usage dump** (`unrecognized arguments: --sample`) |
| Actions artifact `submission-final-35042805806` | the file to upload | **1,183 bytes total** |
| `data/evidence/runs/35042805806/manifest.json` | — | missing (never produced) |

The blend job's own log shows why (`data/evidence/runs/35042805806/blend.log`):

```
pooled shaping: t0=0.372 thin=True -> mean held-out DTI 0.1927 (unshaped 0.1560)
rasterio._err.CPLE_AppDefinedError: _TIFFVSetField:submission.tif: Bad value 3292 for "TileWidth" tag
rasterio.errors.RasterBlockError: The height and width of TIFF dataset blocks must be multiples of 16
```

Three independent defects, all fixed this session and each covered by a regression test:

1. **The writer could not write.** `scripts/blend_submission.py` copied
   `sample_submission.tif`'s rasterio profile — a *striped* GeoTIFF, so `blockxsize == width == 3292` —
   and then forced `TILED="YES"`. GDAL rejects a 3292-px tile width. Fixed by the new
   `src/submission_io.py` (`clean_profile()` drops stale block geometry and pins legal 256-px tiles;
   `write_submission()` **reads the bytes back** and raises unless they are a single-band float32
   raster on the requested grid with non-zero mass). `tests/test_submission_writer.py` reproduces the
   original crash against the old profile, so the defect stays falsifiable.
2. **The failure was masked.** Every workflow step piped Python into `tee` without `pipefail`, so a
   crashed step exited 0 and everything downstream — including "Validate submission format" — "passed".
   `set -uo pipefail` is now on the train / inference / blend / validate steps, and the commit step
   refuses to commit a submission unless it is >10 kB, reads back as a raster and is non-empty;
   otherwise it commits `FAILED.json` instead of a stub.
3. **The binding floor search still used the old grid.** PR #9 replaced `linspace(0.02, 0.9, n)` with
   the log-spaced `shaping_thresholds()` in `src/train.py` — but not in `scripts/blend_submission.py`,
   which is where the 6-fold workflow's *binding* calibration happens. Fixed; measured effect on the
   flat-map case that used to collapse: the floor `t0 = 0.000` is now reachable and a valid sparse
   submission is written instead of an abort.

**Lesson recorded in the site and here:** a workflow conclusion is not evidence. The evidence is the
bytes, read back.

## 2. What this session verified: the scoring universe (and why it changes the target)

The canonical rules PDF was re-read line by line and the relevant sentences are now **machine-verified
verbatim** rather than paraphrased. Actions run **35132263421** (55 s) confirmed all three links in the
chain at once: the canonical PDF at `docs.nlr.gov` is **byte-identical** to the data-tab mirror we
inventoried (`sha256 50d854b1e0239fe6b9648d9fa5c7537bc7b6e5bc10cf6b37a2ff9aa401c36938`, 455,140 B,
`identical: true`), **11/11 quoted sentences matched verbatim**, and a re-fetch of all 76 catalog URLs
recorded 54 `OK_200`, 8 redirects, 10 publisher bot-blocks, 1 login-walled (the data tab) and 2 items to
review. Mechanism: `scripts/verify_rules_quotes.py` extracts the PDF from
`https://docs.nlr.gov/docs/fy26osti/96647.pdf`, normalises whitespace/unicode, and asserts each quoted
sentence appears; the *Verify official sources* workflow also checks that PDF's sha256 against the
copy inventoried from the data tab (`GEMS_96647.pdf`, 455,140 B), proving the mirror and the canonical
document are the same file. Evidence: `data/evidence/rules_quotes.json`,
`data/evidence/rule_sources_verification.json`.

The three load-bearing sentences:

* §1.1 — "In Phase 1, submissions will be evaluated against a privately withheld subset of the original
  **new** fault dataset compiled by expert reviewers."
* §1.1 — "Submissions will be reevaluated against the full, revised **new** fault dataset using the same
  distance-weighted Tversky index."
* §3.3 — "The training labels contain **existing** fault data at 100-m resolution … obtained from the
  INGENIOUS project's Great Basin Regional Dataset Compilation."

So **both prize phases score faults that are absent from the only labels we can download**, and Phase 2
does *not* add the existing catalogue back in. Session 3's phrasing — "the public `existing_faults.tif`
is training data, NOT the scoring target … copying the known-fault raster … is a trap" — was right; what
was missing is the consequence: *every* selection signal in the pipeline (early stopping, fold weights,
and the floor/thinning calibration) is fitted to the catalogue, i.e. to a population that is disjoint
from the scored one. Full write-up and the experiment queue: **`docs/DISCOVERY_PLAN.md`**.

New capability that follows from it — `src/discovery.py` (+ `tests/test_discovery.py`, 6 tests): every
blend report now carries `novel_fraction` (share of emitted probability mass farther than R=3 px from
any catalogued fault), `candidate_new_faults` (connected components that touch no catalogued fault, with
area and extent) and `catalog_recall_R`. Those are the only numbers in the repo that can distinguish
"this model is restating the catalogue" from "this model is proposing faults the catalogue does not
contain" without access to the hidden labels.

Independent corroboration that this is the right frame, from the paper the competition's own About page
recommends (Hermant, Kiersnowski & Bellanger 2025, Stanford SGW, fetched 2026-09-16): they keep **only
fault-bearing training tiles**, explicitly to "limit the possibility of learning images without mapped
faults when there should be some due to **operator observation bias**", and they measure up to **400 m**
of disagreement between the USGS Quaternary catalogue and their own expert labels in north-central
Nevada. Our config keeps 35 % empty windows (`neg_fraction: 0.35`) — a concrete, citable A/B target.

## 3. Verified environment facts (measured, not assumed)

* Sandbox egress allowlist still permits only `github.com`, `api.github.com`, `codeload.github.com`,
  `pypi.org`/`files.pythonhosted.org`. Re-measured 2026-09-16: `www.dropbox.com`, `drivendata.org`,
  `s3.amazonaws.com`, `prd-tnm.s3.amazonaws.com`, `sciencebase.gov`, `apps.nationalmap.gov`,
  `gdr.openei.org`, `raw.githubusercontent.com`, `objects.githubusercontent.com`,
  `pipelines.actions.githubusercontent.com` all fail (TLS EOF).
* **GitHub Actions *artifacts* remain unreachable from the sandbox** — newly measured: downloading
  `submission-final-35042805806` fails with `EOF` against
  `productionresultssa8.blob.core.windows.net`. A *runner* can download them (that is what the new
  re-blend workflow does).
* `github.com` git transport works, so small artifacts (reports, `submission.tif` ≈ 1 MB) travel back
  as ordinary commits. Large rasters cannot and are not committed.
* The full test suite now runs **in-sandbox with torch installed** (CPU wheel from PyPI):
  **37/37 pass** (`pytest tests -q`). Previous sessions could not install torch here at all.

## 4. What was rebuilt this session — and it produced the first real submission

`.github/workflows/reblend.yml` (+ trigger `.github/triggers/reblend`) re-runs **only** the blend step
against the fold artifacts of run 35042805806 — six checkpoints and six full-raster probability maps
(≈120 MB each) that the runner can still fetch, valid until ~2026-09-30. Cross-run artifact download
(`actions/download-artifact@v4` with `run-id`) is the mechanism. Actions run **35131318033** finished in
**7 m 46 s** (the training that it reuses took 2 h 36 min) and committed:

| output | value |
|---|---|
| `data/evidence/runs/35042805806/submission.tif` | **386,018 B**, tiled, blocks 256×256, sha256 `a5ae61d5…` |
| grid / format | 3292×3730, 1 band float32, EPSG:32611, 100 m, NaN outside the footprint (57.93 % of pixels), values in [0,1] |
| emitted area | 21,492 non-zero px of 5,165,852 finite px (0.42 %) |
| validation | `✅ Validation PASSED` — CRS, resolution, dtype, range, size and transform all match the template (`validation.log`) |
| pooled shaping | `t0 = 0.470`, `thin = True`; mean held-out DTI 0.1903 (unshaped 0.1560) |
| informational DTI vs the **known** catalogue | 0.1387 (blanket-ones floor on this grid: 0.0246) — wrong-universe number |
| **discovery profile** | `novel_fraction = 0.684`; 6,090 px in **617** components that never come within R of a catalogued fault (largest 92 px, ≈8.4 km long); catalogue recall @R = 0.214 |

The discovery profile is the interesting one: **68 % of the emitted probability mass sits farther than
300 m from any catalogued fault**, so this submission is not merely restating the catalogue — it is
proposing 617 candidate fault segments the catalogue does not contain. Whether they are real is exactly
what Phase 2's expert review decides, and what the public leaderboard would score; neither is available
here.

**Fold spread is wide and worth acting on** (measured, from the same report): held-out DTI per fold was
0.3351, 0.2665, 0.1780, 0.1776, 0.0916, 0.0742 — a 4.5× spread between the best and worst fold of the
*same* configuration. Equal averaging squeezes a weak fold into a strong one, and the pooled calibration
that picks the floor scores the same folds it fits, so its mean is selection-optimistic. Both are now
instrumented (`scripts/blend_submission.py`): `--calibrate loo` re-fits the floor — and the fold-weight
rule — on the five folds that are *not* being scored, and reports the oracle ceiling next to it, so the
optimism is a printed number instead of an assumption. Experiment `data/evidence/runs/35042805806-dilate-ab/`.

**The width of the emitted line was the binding constraint, not the floor** (measured 2026-09-16,
`data/evidence/shift_robustness.json`, script `scripts/measure_shift_robustness.py`). On the one window
where a written submission and the official label raster coexist (rows 2048–2560, cols 1280–1792 of the
full grid; `data/fixture/fixture_labels.tif`, 5,154 label px):

| band grown around the skeleton | pixels kept | DTI (no shift) | DTI, labels shifted ±3 px |
|---|---|---|---|
| 0 px (what the pipeline wrote) | 801 | 0.0555 | 0.0406 |
| 3 px | 8,063 | 0.1109 | 0.0991 |
| 6 px | 19,908 | **0.1260** | 0.1212 |
| 8 px | 29,249 | 0.1249 | 0.1240 |

The 6-fold submission written on 2026-09-16 kept 801 px in this window against 5,154 label px: it is
**under-covering**, and no floor can fix that — the old search could only choose between a skeleton and
an un-thinned blob. `--dilate-grid` now searches the band width, and the reblend workflow is re-running
the same six saved folds with it (`data/evidence/runs/35042805806-dilate-ab/`, triggered 2026-09-16).

**Reading of the numbers, stated plainly.** The labels in that table are the *public catalogue*, not the
scored set, and translating them is a stress test of the writing operator — not a measurement of the
competition metric. It is reported because the asymmetry is structural: TP<sub>w</sub> credits a
prediction up to R = 3 px away in full, a missed label costs 0.8 per unit, a false positive 0.2. For the
scored faults — new to the expert-reviewed dataset (rules §1.1/§3.5), never seen in training — the
model's localisation error is strictly larger than on the catalogue, which is exactly the regime where a
skeleton loses everything and a band still collects credit.

## 5. Still needed (unchanged by any of the above)

1. **A DrivenData account + enrolment** to (a) download the official data-tab files directly, (b) upload
   submissions at all, and (c) see the public leaderboard — the only unbiased signal for the scored
   universe, 3 submissions/week (rules §3.2). Nothing in this repository can substitute for it.
2. **A GPU** for the full `configs/config.yaml` run (EfficientNet-B5, 10 splits, 60 epochs). CPU runners
   cap the ensemble at resnet34 / 45 epochs / ~2.5 h per fold, which is what run 35042805806 measured.
3. **Eligibility check** (rules §1.3): competitors must be US citizens/permanent residents, private
   entities US-incorporated, academic institutions US-accredited; FFRDC-affiliated researchers may
   compete individually but are not cash-eligible; DOE employees and support contractors are excluded.
4. **Narrative + code assets** (rules §3.2, §3.5): finalists must submit complete code assets and the
   generative-AI disclosure — this repository is structured to be that submission (manifests,
   environment pins, reproduce page, audit gates).
5. **1 m DEM derivatives** (716 tiles already confirmed against the live USGS 3DEP bucket) and the
   proxy-catalogue evaluation of `docs/DISCOVERY_PLAN.md` §3.
6. **The public leaderboard score itself.** Every number in this repository is a proxy: catalogue DTI is
   the wrong universe, discovery diagnostics are unlabelled, and the width study is a stress test. Three
   submissions a week against the real metric (rules §3.2) is the only way to replace proxies with
   measurements, and it needs the account from item 1.

---

## 6. Session 5 (same day): the rules verified verbatim, and the emission width

### 6.1 The scored population, now machine-verified rather than paraphrased

Session 4 established that both prize phases score expert-mapped *new* faults. That conclusion has now
been re-derived from the official PDF on a runner, sentence by sentence, by
`scripts/verify_rules_quotes.py` (workflow `verify-rules.yml`, run
[35134058898](https://github.com/buffedlizard55-lab/GEMSDOE/actions/runs/35134058898), 16 s):

* **19 of 19 quoted sentences matched verbatim** after NFKC + quote folding + de-hyphenation +
  whitespace collapse. No fuzzy matching; a miss fails the step and the commit.
* The document fetched from `docs.nlr.gov/docs/fy26osti/96647.pdf` is **byte-identical** to the data-tab
  mirror recorded in `data/evidence/inventory.json`: `sha256 50d854b1e0239fe6b9648d9fa5c7537bc7b6e5bc10cf6b37a2ff9aa401c36938`,
  455,140 B, `identical: true`.
* Evidence: `data/evidence/rules_quotes.json` (+ `rules_quotes.log`), committed by the workflow; the home
  page of the site renders the six decisive sentences with their found/not-found marks.

The sentences that carry the strategy, quoted exactly:

| id | § | sentence |
|---|---|---|
| `phase1_target` | 1.1 | "In Phase 1, submissions will be evaluated against a privately withheld subset of the original new fault dataset compiled by expert reviewers." |
| `phase2_target` | 1.1 | "Submissions will be reevaluated against the full, revised new fault dataset using the same distance-weighted Tversky index." |
| `phase2_eligibility` | 1.1 | "All Phase 1 competitors will be eligible to compete in Phase 2 and will be automatically submitted for consideration." |
| `experts_revise` | 1.1 | "After Phase 1, expert reviewers will use submitted predictions to revise the new fault dataset." |
| `labels_source` | 2 | "The labels for this prize come from the USGS Quaternary Fault and Fold Database and from a set of newly identified faults labeled by geology experts at the National Laboratory of the Rockies (NLR) and USGS." |
| `ranking_basis` | 3.2 | "Second-round prize rankings will be determined by running the selected final submissions against the complete updated test set created by expert review." |

Two consequences worth stating because they are easy to get backwards: the released training label
raster is the *existing* catalogue (problem page + rules §3.3) while both phases score the *new* faults,
so reproducing the catalogue earns credit only where the experts' new labels coincide with it; and no
Phase-1 top-5 cutoff gated Phase-2 entry, so there is nothing to be gained by trading discovery for
catalogue coverage.

### 6.2 The emission width was the binding constraint, not the floor

Measured (`data/evidence/shift_robustness.json`, script `scripts/measure_shift_robustness.py`) on the one
window where a written submission and the official label raster coexist — full-grid rows 2048–2560 ×
cols 1280–1792, 5,154 label pixels:

| band grown around the skeleton | pixels kept | DTI (labels as-is) | DTI (labels shifted ±3 px) |
|---|---|---|---|
| 0 px — what the pipeline wrote | 801 | 0.0555 | 0.0406 |
| 3 px | 8,063 | 0.1109 | 0.0991 |
| 6 px | 19,908 | **0.1260** | 0.1212 |
| 8 px | 29,249 | 0.1249 | 0.1240 |

The submission kept 801 px against 5,154 label px in that window: it is **under-covering**, and the old
search could not express the fix — its only two options were a pure skeleton and the un-thinned
floor-passing blob. `scripts/blend_submission.py --dilate-grid` (default `0,1,2,3,4,6`) now searches the
band width on the held-out crops, and `--calibrate loo` re-fits the floor — and the fold-weight rule — on
the folds that are *not* being scored, printing the oracle ceiling next to the honest mean, so the
selection optimism is a number instead of an assumption.

Honest boundary: the labels used in that table are the public catalogue, not the scored set, and shifting
them is a stress test of the writing operator. It is reported because the asymmetry is structural
(TP<sub>w</sub> credits a prediction up to R = 3 px away in full; a missed label costs 0.8, a false
positive 0.2), and because it is *most* severe exactly where the scored faults live — faults the model
never saw in training, whose traces it cannot localise as tightly as the catalogue's.

Runner A/B on the six saved folds (emission width + LOO audit, no retraining):
`data/evidence/runs/35042805806-dilate-ab/` — workflow `reblend.yml`, run 35133590776. Its verdict decides
the default: a wider band is kept only if the LOO mean beats the skeleton by > 0.01.
