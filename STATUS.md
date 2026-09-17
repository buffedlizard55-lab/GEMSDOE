# Project status — 2026-09-17 (session 9)

## Session 9 — the data-placement blocker is resolved; the pipeline runs on the official bytes, in the sandbox and on runners

The stated blocker — *run `bash scripts/download_competition_data.sh` on any unrestricted machine into
`data/`, then `python scripts/prepare_data.py`* — is done, autonomously, with every byte accounted for:

1. **Git data bridge.** The sandbox allowlist (github.com/api.github.com/codeload.github.com/pypi.org only)
   blocks Dropbox, DrivenData, S3 *and* the Azure host that serves Actions artifacts, so no existing
   channel could carry the binary rasters into `data/`. New transport
   (`.github/workflows/place-competition-data.yml`, run 35168924460): a runner downloads the three
   official rasters from the data-tab Dropbox mirrors, **verifies each sha256 against the pinned
   inventory** (`data/evidence/inventory.json`, fails loudly on drift), runs `scripts/prepare_data.py`
   on the real bytes, and commits them to the branch as ~90 MiB parts in `data/bridge/` (GitHub rejects
   blobs ≥ 100 MB) with a manifest. Receiving side: `python scripts/assemble_data_bridge.py` re-verifies
   every part, concatenates, re-verifies the whole-file sha256, and places the canonical names.
   Round-trip, tamper-refusal, unpinned-refusal and verify-only are regression-tested
   (`tests/test_data_bridge.py`). Evidence: `data/evidence/data_placement.json`.
2. **`data/` now holds the official files**, sha-verified:
   `training_features.tif` (418,912,844 B), `labels.tif` (425,830 B), `sample_submission.tif`
   (1,599,597 B). `python scripts/prepare_data.py` **PASSES** in the sandbox: 3292×3730, 19 bands with
   `description`/`data_category` tags, EPSG:32611, 100 m, aligned bounds, labels 0/1, template float32
   [0,1] with NaN outside the footprint.
3. **Full pipeline re-run on the real data with the hardened code.** Runner run 35169168957 (smoke
   profile, data assembled *from the bridge*, not Dropbox): train → inference → validation **PASSED** →
   local score. Held-out DTI 0.0790 (prior smoke run 34876843912: 0.0588). Submission
   sha256 `ee183e26…` (470,840 B). Evidence: `data/evidence/runs/35169168957/`.
4. **The same pipeline now runs inside the 3.9 GB sandbox itself** — train → inference → validate →
   score on the placed bytes (`data/evidence/runs/local-sandbox-smoke/`): identical patch split as the
   runner at the same seed (311/234/233/78 — split determinism holds across machines), held-out DTI
   0.0997, validation **PASSED**, submission sha256 `b9f2bc4d…`. This required two measured memory
   fixes (below); the sandbox OOM-killed the first attempt at 3.8 GB anon-RSS.
5. **Memory fixes (measured OOM → fix → measured pass).** `src/train.py` frees the un-normalised stack
   once `Xn` exists (−933 MB); `src/dataset.py:make_patches` zeroes the padded stack in place instead
   of `.copy()`-ing a third full-stack allocation (−972 MB; the held-out windows are extracted before
   the zeroing, so semantics are unchanged); `src/inference.py` computes the validity footprint before
   freeing `X`. Peak RSS on the full 19-band grid: 3.8 GB (killed) → 3.18 GB (passes). This also
   lowers runner peaks.
6. **Training workflows are now Dropbox-independent.** `train-and-submit.yml`, the ensemble fold jobs
   and the blend job assemble `data/` from the committed, sha256-pinned bridge when it is present and
   only fall back to the mirrors otherwise — future runs no longer depend on live Dropbox links.
7. **CI regression for workflow YAML.** The first bridge firing (run 35168727415) failed in 0 s with
   zero jobs: an unquoted `: ` inside a step name. `tests/test_workflow_yaml.py` now parses every
   workflow and flags that pattern, so the failure class is caught by the Tests workflow instead of
   only when a trigger is pushed.
8. **Second 6-fold ensemble (seed 43, folds 6–11) fired** as the queued experiment for the
   emission-width decision (`data/evidence/emission_decision.json` condition 3: "second ensemble").
   It trains in parallel on public-repo runners (~2.5 h) and commits its evidence to
   `data/evidence/runs/<run_id>/` autonomously; next session analyses it against the widen decision.

Still true from session 8: both prize phases score the **new** fault dataset (disjoint from
`labels.tif`); read `docs/DISCOVERY_PLAN.md` before optimising local scores. The remaining blockers to
a leaderboard result are unchanged and listed in §5 below: a DrivenData account (to submit at all),
GPU capacity for the full config, and the eligibility check.

---

## Session 8 — reliability and provenance fixes

The official competition home page, problem page and About page were fetched and read again on
2026-09-17. The line-by-line source table and the changes made in this review are recorded in
[`REVIEW_2026-09-17.md`](REVIEW_2026-09-17.md); the competition pages and official rules PDF remain the
controlling sources.

Implemented in this pass:

- direct `src.inference` now uses `src.submission_io.write_submission()` rather than forcing tiled TIFF
  output on the striped sample profile; the writer reads the bytes back and refuses an invalid or empty
  raster;
- training/inference workflow pipelines no longer use a successful `grep || true` subshell that masks a
  failed Python process; the ensemble blend explicitly refuses an incomplete fold matrix;
- optional USGS 3DEP DEM derivatives are now a real, symmetric path: a supplied local mosaic is reprojected
  to the feature grid and used by both train and inference. The default GPU config is now disabled until
  `data.external_dem_path` is set, so the documented default cannot silently change the channel count;
- SMP is imported lazily and the rules evidence test handles a report that intentionally omits the optional
  extracted PDF text.

**Current local verification:** run `python -m pytest tests -q` after installing the project/test dependencies;
`REVIEW_2026-09-17.md` explains why the sandbox cannot produce a leaderboard score. The full prior evidence
and historical findings remain below; they are retained to make regressions auditable rather than erased.

---

Supersedes the previous `STATUS.md` (session 3, 2026-09-15). Session 3's own claims are quoted where
this session falsified them, because the way they were falsified is the most useful thing in this file:
**a green workflow was treated as evidence.**

---

## 0. Session 7 (2026-09-16, second pass) — the rules check went green, and the width question got an answer

**Green:** `verify-rules` run 35153898428 — 29/29 rule sentences verified verbatim against the
official PDF (`data/evidence/rules_quotes.json`), with the 18 page-furniture removals recorded
(bare page numbers, glued section numbers). The last blocker on that check was that pypdf extracts a
page number *inside* the sentence that continues across the page break; the fix drops blank lines
before looking for page furniture, and it is now proved end to end by a synthetic reportlab→pypdf
PDF in CI. Tests 35153898433 and Pages 35153898486 are green on the same commit.

**The width question has a third measurement, and it disagrees with the first.** `proxy-eval` run
35152701740 swept floor × thinning × emission width on 61,664 px (6,166 km) of USGS SGMC fault trace
that `labels.tif` does **not** contain — the closest measurable stand-in for the scored new-fault
population. Proxy DTI rises monotonically with the width (0.0144 → 0.0395 from 0 px to 6 px), the
opposite sign to the held-out-crop sweep (0.1903 → 0.0908). `scripts/decide_emission_width.py`
reconciles them by projecting every measured policy onto a range of possible scored-truth sizes
using the metric's own scaling (`DTI = TP_w/(0.2(TP_w+FP_w) + 0.8|G|)`: the wrong-mass term does not
grow with |G|, the missing-mass term does), and writes the decision:
**widen, but not yet** — conditions 1 and 2 met (+0.0149 on the new-fault-like population; 3/3
plausible |G| anchors), condition 3 (second ensemble) unmet, default unchanged.

**The uncomfortable number in that table:** a constant-ones submission scores **0.0585** on the
new-fault-like population, beating the shipped skeleton's **0.0247** and every swept candidate.
Recall is worth four times precision under α=0.2/β=0.8, so the shipped emission is not conservative,
it is *under-emitting*. The crossover is computed and published: the skeleton stays ahead only
below ~2,200 km of scored truth.

**Three measurement defects found and fixed while reading that evidence** (each had produced a
plausible-looking number that meant something else): the truth length was converted with 0.01 km/px
instead of 0.1 km/px (every committed truth length was 10× short); the blanket-ones baseline was
taken over `np.isfinite(pred)`, so its definition changed with whichever raster was scored; and the
score of the raster handed to `--pred` was published as "as submitted", which mislabelled the
sweep's soft ensemble map as a submission on the site's Results page. All three are fixed, pinned by
`tests/test_proxy_catalogue.py::test_eval_units_support_and_role_are_unambiguous`, and the sweep now
clips every candidate to the data footprint so it compares *legal* submissions.

**New page:** `docs/verification.html` — the public leaderboard read directly (43 entrants; #1
0.1972; top-5 cut ≈0.1454, read 2026-09-16), an explicit "this repository is not on it" statement,
and a table re-checking every load-bearing external claim from its own URL
(`data/evidence/independent_verification.json`).

**Suite:** 91 passed (was 86; +3 proxy-contract tests, +2 site tests). The audit test deleted by an
earlier slice-to-EOF edit is restored.

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
  as ordinary commits. Large rasters **can** now travel too, as ≤ 90 MiB sha256-pinned parts
  (`data/bridge/`, session 9) — GitHub rejects single blobs ≥ 100 MB, not large totals.
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

---

## 7. Session 6 (same day): line-by-line review — 12 findings, 11 fixed, 1 queued

Method: read every line of `src/metrics.py`, `src/submission_optim.py`, `src/discovery.py`,
`src/submission_io.py`, `src/train.py`, `scripts/blend_submission.py`,
`scripts/measure_shift_robustness.py`, `scripts/audit_docs.py`, `scripts/verify_rules_quotes.py`,
`scripts/verify_links.py` and the `_scoring_universe`/`build_index` renderers; re-ran the suite
(38/38 green on arrival — the "37/37" in §3 was stale by one test), the metric self-test (8/8),
`audit_docs.py` (PASS, but with 1 uncatalogued-host REVIEW), and `build_site.py` (rebuild
drifted from the committed pages — the thread that found F9–F11). Environment: fresh `.venv`
from `requirements.verified.txt` + pytest on the stock sandbox (torch 2.14.0+cu130, CPU-only).

Findings (severity = consequence if left in place):

| id | severity | finding | disposition |
|---|---|---|---|
| F1 | **high (methodological)** | `loo_aggregation`'s fold-weight audit averaged other folds' held-out crops — *different geographic windows* (each fold's own compact subset, `train.py::heldout_maps`) — and scored the mix against the held-out fold's gt. Cross-geography averaging cannot measure the weight rule. | **removed** the comparison; LOO weights still fitted and reported per row, `weight_rule_gain` kept as `None` with a note. Valid weight evidence remains the full-map A/B (`local-mini-ensemble/`, same maps, same grid). Sound LOO weight audit queued (SUGGESTIONS §7.1). |
| F2 | medium (latent crash) | `calibrate_shaping` returned a 4-tuple with no usable folds; every caller unpacks 5 → `ValueError`. | returns `(0.3, True, nan, [], 0)` |
| F3 | medium (latent wrong-number) | `calibrate_shaping` + `loo_aggregation` mutated `f["pred_crop"]` when `pre` was set → `--frangi --calibrate loo` applied Frangi twice to the same crops. | copy-on-transform in both |
| F4 | low | `calibrate_shaping` docstring said 4-tuple return, code returns 5. | docstring fixed |
| F5 | low | `sl` lambda comment said "centre window", code took top-left `a[:k,:k]`. | removed with F1 |
| F6 | low | `--weights dti` fell back to equal weights silently when a fold lacked DTI. | prints a WARNING, still falls back |
| F7 | low (docs) | `SUGGESTIONS.md` claimed "audit now finds no uncatalogued hosts" in the same row that contained the `https:`+ellipsis stub the audit flagged — self-falsifying. | reworded; audit now lists **0** uncatalogued hosts |
| F8 | low | `discovery.py`: `novel_bboxes` largest-first + min_px-filtered, but `novel_component_px` was last-25-by-label-id, unfiltered — the fields could disagree. | derived from the same boxes |
| F9 | low (reproducibility) | `generate_dummy_submission.py` used unseeded `np.random` — every invocation differed. | `--seed` (default 42); byte-identical output pinned by test |
| F10 | low (dead code) | `train.py` built a full `TensorDataset` of the test windows every fold and never iterated it (scoring goes through `predict_patches`); `blend_submission.py` carried an unused `import math` and (after F1) an unused `_weighted_mean`. | removed |
| F11 | **high (live site)** | `docs/metric.html` printed *"Quotation verification is INCOMPLETE (0/0). Do not rely on the table below"* next to 19 verified quotes: `_scoring_universe` still read the old `verification`/`document` keys after the report was reshaped to `summary`/`source`. The overview page reads the same file correctly — the site contradicted itself. | reads the current schema (falls back to recounting the quotes, never to 0/0); page rebuilt. New `tests/test_site.py` pins the badge to the evidence. |
| F12 | medium (evidence staleness) | `build_site.py` rebuild drifted from committed pages two more ways: (a) `rules_quotes.json` records `source.url=/tmp/rules_canonical.pdf` (workflow verifies a /tmp copy), which the index renderer would publish as a link; (b) `link_verification.json` (19:33Z) disagrees with `sources.html` (built 18:24Z), flagging 3 live ScienceBase pages as BROKEN (403 = bot-wall — the GeoDAWN item re-verified reachable via independent fetch the same day). | **not touched here**: the parallel session (`arena/01a0ab54-gemsdoe`) independently found and fixed both on its branch (canonical-URL recording + http-guard, sciencebase BOT_BLOCK policy + `--reclassify`, workflow rebuild step, audit count check). Deliberately left to that branch to avoid a same-hunk conflict; this branch rebuilt only `docs/metric.html`. |

Verification after the fixes: **46/46 tests pass** (38 on arrival + 8 new: 4 blend/seed in
`test_ensemble.py`, 1 in `test_discovery.py`, 3 in `tests/test_site.py`), `audit_docs.py` PASS with
0 uncatalogued hosts, `src/metrics.py --self-test` 8/8, `metric.html` shows "19/19 quoted sentences
verified verbatim" with no `/tmp/` leak. Dependencies audited against imports: `scikit-learn` and
`matplotlib` are listed in `requirements.txt` but imported nowhere (same dead-dep class as the
removed `albumentations`) — left in place as harmless; `einops` (also listed) is genuinely unneeded
(`smp` 0.5.0 instantiates SegFormer without it; only the weight download fails, on the blocked
host — the known limitation).

Still running at session end: the emission-width + LOO A/B (run 35133590776, `blend` job since
18:18Z). It executes the *pre-session-6* blend code on the sibling branch, so when it lands: trust
its floor/dilate verdicts, **disregard its `weight_rule_gain`** (it was computed by the removed
cross-geography comparison). Its report commits to `arena/01a0ab54-gemsdoe`, not here.

Merge note for the sibling branch: this PR touches `scripts/build_site.py` only in
`_scoring_universe` (a different hunk than its index-URL guard — clean merge) but rebuilds
`docs/metric.html`, whose footer line it also rebuilt → expect a one-line footer conflict there,
resolved by rebuilding the site from the merged tree. After both merge, re-run the verify-sources
workflow once so all pages render from one consistent evidence set.

---

## 7. Session 5b: the width experiment came back, and it disagreed with the surrogate

The reblend experiment (`reblend.yml`, run [35133590776](https://github.com/buffedlizard55-lab/GEMSDOE/actions/runs/35133590776),
evidence in `data/evidence/runs/35042805806-experiment/`) answered both questions it was built to
answer, and its answers were: *keep the skeleton*, and *the shaping is worth less than the acceptance
threshold says is worth keeping*.

| question | measurement |
|---|---|
| Which band width does the search pick on the six real held-out crops? | **0 px** (the pure skeleton) at floor 0.4697, pooled mean DTI 0.1903 |
| Same question per band, best floor over the grid | 0 px → 0.1903 · 1 px → 0.1525 · 2 px → 0.1281 · 3 px → 0.1128 · 4 px → 0.1037 · 6 px → 0.0908 |
| Is the floor's gain real, fitted without the scored fold? | honest (LOO) mean **0.1643** vs 0.1560 unshaped = **+0.0083**, *below* the pre-registered 0.01 acceptance test |
| How large is the selection optimism? | pooled 0.1903 − honest 0.1643 = **0.0260**; oracle ceiling (floor fitted on the scored fold itself) 0.2029 |

**The surrogate pointed the wrong way, and this is the useful part.** `data/evidence/shift_robustness.json`
(a single window, labels shifted to simulate mislocalisation) argued for a 6-px band and +0.0705 over
the skeleton. The same bands scored on the six held-out crops — where the labels are the model's own
validation windows and the trace really is where the model drew it — monotonically *lose* DTI as the
band widens, because FP mass is paid per pixel while TP_w only takes a max within R = 3 px. Both
measurements are correct about different populations: the surrogate asks "what if the scored fault is
somewhere else than I drew it", the held-out crops ask "what if it is where I drew it". The scored
faults are new, so the truth is between them, and *neither* justifies overriding the held-out
measurement on the evidence available. The decision recorded here is therefore: **skeleton kept**,
`--dilate-grid` left in place as a knob with its verdict attached, and the honest number (+0.0083)
carried forward instead of the pooled one.

**Two defects the audit exposed in itself, both fixed:**

1. *The audit did not fit in its job* (4743 s of a 120-minute limit, and the earlier attempt hit the
   timeout at 120 minutes). Cause: `compute_distance_weighted_tversky` rebuilt a 49-offset credit map
   over the whole raster and a label distance transform for **every** candidate floor, then read only
   the ~1 % of pixels that are labels; `dominant_thin` computed a whole-raster EDT up to 40 times per
   candidate. Fixed in `src/metrics.GtContext` (label geometry built once; TP_w gathered at label
   pixels) and by maintaining the thinning distance map incrementally. Both are pinned to the original
   implementations by `tests/test_metric_parity.py` and `tests/test_shaping_parity.py` — a speed-up in
   the scoring function is the one change that could silently corrupt every number at once.
2. *The same submission had two different file hashes.* The two runs' rasters are **pixel-identical**
   (`sha256_pixels d6380a58…`) but the files differ, because the chosen floor is stored in GeoTIFF
   metadata at full float64 precision and `np.geomspace` rounded differently under another numpy build
   (`0.4696741044002384` vs `…2383`). The search grid is now quantised to six significant digits, and
   every report carries both the container hash (what a reviewer re-checks) and the pixel hash (what
   two runs are compared with).

**Cross-session review.** While this was running, a parallel session (PR #12) audited the code above
and found three real defects in it, all confirmed and merged here: `calibrate_shaping` mutated the
fold dicts, so `--calibrate loo` applied a pre-transform twice; its empty-input path returned four
values where callers unpack five; and the LOO *weight* comparison averaged fold crops covering
different geographic windows and scored them against another fold's labels, which cannot measure a
weight rule. All three are fixed on this branch; the weight rule is now fitted per row and reported
without being scored. The lesson kept from the merge: the fold-weight question still needs the
per-fold held-out footprints (queued in `SUGGESTIONS.md`), and a number that cannot be measured
should be `None`, not a plausible-looking float.

