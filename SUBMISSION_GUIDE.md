# Submission Guide — GEMS Prize (Verified from Official Sources)

**Primary sources:**
- Problem description submission format: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
- Official Rules PDF: https://docs.nlr.gov/docs/fy26osti/96647.pdf (Sections 3.2, 3.3, 3.5, 3.6)
- Competition main: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution

## 1. What to Submit (Per Problem Page)

> For this competition, you will submit a GeoTIFF file containing your predictions for **all** faults in the region.

Requirements (line-by-line verified from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format):

- [ ] Same projected CRS as training data: **UTM zone 11N, EPSG:32611** (https://epsg.io/32611)
- [ ] Same resolution: **100m**
- [ ] Same bounds as training data, data outside bounds is null or NaN
- [ ] Single layer, datatype **32-bit float (float32)**, values **between 0 and 1** indicating confidence/probability, higher = higher probability
- [ ] Sample submission that predicts total fault absence is provided for reference on data download page (https://www.drivendata.org/competitions/306/competition-doe-gems/data/)

## 2. How to Enter (Per PDF Section 3.1, 3.2)

From PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf:

- [ ] Create profile on DrivenData platform and agree to competition rules and restrictions
- [ ] Navigate to challenge website to sign up as competitor: https://www.drivendata.org/competitions/306/competition-doe-gems/
- [ ] Review competition materials and data (problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ and about https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/)
- [ ] Download training features and labels from data tab https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (requires login)
- [ ] Training dataset: geophysical features for GeoDAWN study area at 100m resolution, multiband GeoTIFF, one feature per band. Plus instructions for downloading USGS DEM at 1m.
- [ ] Training labels: existing fault data at 100m resolution where positively labeled pixels indicate fault presence, obtained from INGENIOUS Great Basin Regional Dataset Compilation DOI https://doi.org/10.15121/1881483 (PDF footnote 4)
- [ ] Competitors will submit predictions for all faults in GeoDAWN study area as GeoTIFF raster at 100m resolution. Example valid submission provided.
- [ ] **Generative AI disclosure:** If using generative AI, indicate in narrative (not included in word count) extent and how used (PDF section 3.2). Responsible for accuracy, authenticity, authorship.

## 3. Feedback & Limits (Per PDF Section 3.4)

- [ ] Each entity may submit **more than one set of predictions** for automated scoring up to **three per week**, as specified on competition website
- [ ] By submission deadline, **must select only one set of predictions** to use as final submission for final evaluation and ranking
- [ ] Multiple finalized submissions not allowed
- [ ] Each entity (team/org/individual) allowed **one final submission**; individuals on team not allowed separate final submission

## 4. Evaluation & Ranking (Per PDF Section 3.6)

- [ ] Using **distance-weighted Tversky index** metric published on competition website, judges score chosen submission against ground truth
- [ ] Ground truth divided into **public test dataset** and **private test dataset**
- [ ] **Public leaderboard:** score on public test dataset shown while competition running
- [ ] **Private leaderboard:** score on private test dataset used for first prize round when competition closes
- [ ] **Second round:** score against complete updated test set created by expert review after close
- [ ] **Must choose only one submission** for scoring across both prize rounds, **without knowledge of private test scores** — to encourage generalization, discourage overfitting to public test (PDF 3.6.2)
- [ ] Set of faults in public test and relative weight determined by organizers before start

### Prize Structure (Per PDF 1.1 and Problem Page)

- **Initial Round $50,000:** Top 5 each $10,000, judged on private test set of fault labels
- **Expert review:** Panel uses submitted predictions to update fault labels for full region
- **Final Round $250,000:** Top 5 judged on all fault labels in updated set — 1st $100k, 2nd $70k, 3rd $40k, 4th $25k, 5th $15k
- Same submission scored twice; predictions that helped experts identify previously-unmapped faults can score higher in final round

## 5. Solution Verification and Delivery (Per PDF 3.2, 3.5)

For finalists, for chosen algorithm, must submit:

- [ ] **Complete code assets and documentation**, including:
  - Description of resources required to build and run solution
  - Assets should be able to sufficiently reproduce winning results and generate predictions on new data samples
- [ ] Documentation consistent with DrivenData's Winning Model Documentation Template (provided to winners after competition)
- [ ] Finalists must sign and return required documents including eligibility certifications
- [ ] Award approvals — Official winners selected by DOE, may take into account program policy factors listed in Appendix A (PDF section 1.3 eligibility, 1.4 prize goals)
- [ ] DOE is judge and final decision maker, may elect to award all, none, or some submissions
- [ ] After winners notified, prize administrator requests necessary information to distribute cash prizes
- [ ] Interviews may be held after announcement (PDF 3.6.3), not required

## 6. How to Submit via DrivenData (Practical)

1. Go to https://www.drivendata.org/competitions/306/competition-doe-gems/
2. Click "Compete!" to enroll (requires account)
3. Download data from https://www.drivendata.org/competitions/306/competition-doe-gems/data/
4. Train model using this repo: `python -m src.train --config configs/config.yaml`
5. Generate predictions: `python -m src.inference --config configs/config.yaml --model-dir outputs --out submission.tif`
6. Validate format: `python scripts/validate_submission.py --pred submission.tif --sample data/sample_submission.tif`
7. On DrivenData, click "Submit" → "Make new submission" → upload GeoTIFF
8. Check public LB score
9. Before deadline **Dec 3, 2026 11:59pm UTC**, select ONE submission as final for both prize rounds
10. If finalist, prepare code + docs per template

## 7. Submission Format Validation (Our Implementation)

`scripts/validate_submission.py` checks:

- CRS == EPSG:32611
- Resolution == 100m
- Bounds match sample submission (or training_features.tif)
- Single band, float32, values in [0,1]
- No NaN outside? Actually NaN allowed outside bounds, but inside should be 0-1
- File size matches expected

See reference solution for example: https://github.com/drivendataorg/gems-prize-reference-solution

## 8. Irregularities Flagged

- Sample submission file not available without login — we create dummy if missing in `scripts/prepare_data.py`
- Training features file naming: problem page says `training_features.tif`, reference solution says `numeric_features.tif` — we handle both in `src/dataset.py`
- 1m DEM links CSV — links likely point to The National Map, but actual download requires handling large tiles — we provide fallback via LidarExplorer https://apps.nationalmap.gov/lidar-explorer/

## 9. No Hallucinations

All requirements above copied line-by-line from official sources, with links for manual verification.
