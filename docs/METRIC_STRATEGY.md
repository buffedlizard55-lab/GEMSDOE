# What the GEMS metric actually rewards — measured, not assumed

**Status:** every number on this page was produced by `scripts/metric_strategy.py` running
against a **real window of the official competition rasters** (`data/fixture/`, a 512×512
crop at row 2048 / col 1280 of `existing_faults.tif`, |G| = 5,154 fault pixels, 1.97 % of
valid area). Reproduce with:

```bash
python scripts/metric_strategy.py            # uses data/fixture
python scripts/metric_strategy.py --labels data/labels.tif   # full raster
```

Metric source (verbatim definition):
<https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric>

---

## 1. The algebra that changes everything

The official definition gives, term by term,

```
TP_w = Σ_{g∈G} max_{x:d(x,g)≤R} p(x)·k(d(x,g))
FN_w = Σ_{g∈G} [1 − max_{x:d(x,g)≤R} p(x)·k(d(x,g))]
```

so **`TP_w + FN_w = |G|` identically, for any prediction whatsoever**. Substituting
`FN_w = |G| − TP_w` into `DTI = TP_w /(TP_w + α·FP_w + β·FN_w)` with α=0.2, β=0.8:

> ### DTI = TP_w / ( 0.2·(TP_w + FP_w) + 0.8·|G| )

`scripts/metric_strategy.py` asserts this rearrangement against the direct computation for
every row below, so the identity is verified, not claimed.

Three consequences follow immediately:

1. **`0.8·|G|` is a constant floor** in the denominator — set entirely by the test set, not
   by you.
2. **`TP_w` and `FP_w` carry the same coefficient (0.2).** A unit of probability mass costs
   the same wherever you put it. It only earns numerator credit if it is the strongest mass
   within R=3 px of a ground-truth pixel.
3. **`TP_w` saturates at 1 per ground-truth pixel.** Extra mass along an already-covered
   fault earns nothing and costs 0.2 per unit.

## 2. Measured baselines (real competition data)

| Strategy | DTI | TP_w | FP_w | mass |
|---|---:|---:|---:|---:|
| all zeros | 0.00000 | 0.0 | 0.0 | 0 |
| **all ones (blanket coverage)** | **0.09560** | 5153.7 | 243768.4 | 261,835 |
| uniform p=0.05 | 0.03897 | 257.7 | 12188.4 | 13,092 |
| uniform p=0.25 | 0.07776 | 1288.4 | 60942.1 | 65,459 |
| uniform p=0.5 | 0.08881 | 2576.8 | 121884.2 | 130,918 |
| exact known faults, p=1 | 0.99993 | 5153.5 | 0.0 | 5,153 |
| known faults dilated r=1 | 0.88182 | 5153.7 | 3452.0 | 15,509 |
| known faults dilated r=2 | 0.73358 | 5153.7 | 9357.2 | 25,816 |
| known faults dilated r=3 | 0.58868 | 5153.7 | 18003.3 | 35,935 |
| known faults dilated r=6 | 0.35842 | 5153.7 | 46125.4 | 64,192 |
| triangular ramp radius 3 px | 0.81180 | 5153.5 | 5971.7 | 18,067 |
| triangular ramp radius 8 px | 0.45040 | 5153.6 | 31441.1 | 47,268 |
| **25 % of known faults, p=1** | **0.55448** | 2571.4 | 0.0 | 1,288 |
| 50 % of known faults, p=1 | 0.79391 | 3891.3 | 0.0 | 2,577 |
| 75 % of known faults, p=1 | 0.91657 | 4627.5 | 0.0 | 3,864 |

## 3. The strategic conclusion (and a correction to the obvious reading)

The naive reading of α=0.2 / β=0.8 is *"false negatives are penalised 4× — so predict
generously."* **The measurement says the opposite.**

- Predicting **everything** scores **0.0956**.
- Predicting only **a quarter** of the faults, but *precisely*, scores **0.5545** — nearly
  **6× better** while emitting 200× less probability mass.

The reason is that β multiplies `FN_w`, which is bounded by `|G|`, whereas α multiplies
`FP_w`, which grows with the **area** you paint. On this window there are 5,154 fault
pixels against 261,835 valid pixels; a blanket prediction buys 5,154 units of TP and pays
for 243,768 units of FP. The 4× per-unit discount on false positives is overwhelmed by the
~50× difference in how many units there are.

> **Operating rule for this competition: be sparse, thin and confident.**
> Spatial *precision* dominates. Dilating perfect predictions by a single pixel already
> costs 12 points of DTI (0.9999 → 0.8818); by 3 px it costs 41 points.

This is why `src/submission_optim.py` implements a probability floor plus
distance-R dominating **thinning**, and why the pooled shaping search in `src/train.py`
consistently selects `thin=True`.

### Corollaries for modelling

- **Skeletonise / non-maximum-suppress along the fault normal** before writing the
  submission. Faults are 1-px lineaments at 100 m; a 3-px-wide confident ridge throws away
  ~40 % of the achievable score.
- **Calibration beats raw discrimination.** Because TP saturates at 1, pushing a correct
  pixel from p=0.6 to p=1.0 gains at most 0.4 of TP but costs 0.4·0.2 of FP — nearly free.
  Pushing a *wrong* pixel the same way is pure loss. Confidence should be concentrated.
- **The blanket-coverage floor (0.0956) is the number to beat.** A model scoring below it
  is worse than a constant. The first CI smoke run scored **0.0816** — i.e. *below* the
  trivial baseline, which is exactly what a 2-epoch MobileNetV2 should do and is recorded
  here as an honest negative result, not a success.

## 4. Caveats (flagged, not hidden)

- These numbers are computed against the **public/known** fault labels. The competition
  scores against a **private set of newly-labelled faults** not in that database
  (<https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure>).
  The *shape* of the metric is identical, so the strategic conclusions transfer; the
  absolute values do not.
- `|G|`, and therefore the `0.8·|G|` floor, is set by the hidden test set and is unknown.
- ε in the denominator is not specified on the problem page; we use 1e-7. Irrelevant at
  these magnitudes but recorded for exactness.
- The 512×512 fixture is one window. Run `--labels data/labels.tif` on a machine that has
  the full raster to confirm at full scale.
