# Results on ADNI (October 2026 download)

Aggregate results only. No subject-level ADNI data is stored in this repository.

Data used in preparation of this work were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu).

## Data used

| Table | Used for |
|---|---|
| PTDEMOG, MMSE, ADAS, FAQ | Stage 1 core inputs (age derived from `PTDOB`) |
| VITALS, GDSCALE, MEDHIST | Stage 1 comorbidities: BMI, systolic BP, depression (GDS), cardiovascular / endocrine / psychiatric / neurologic / smoking history |
| UPENN_PLASMA_FUJIREBIO_QUANTERIX | Stage 2: plasma p-tau217, Aβ42/40, NfL, GFAP |
| UGOTPTAU181, BLENNOW plasma NfL | Stage 2: p-tau181 and older-assay NfL (covers ADNI1/GO/2 participants) |
| APOERES | APOE ε4 allele count (genotype; available in every stage) |
| UCSFFSX7 (FreeSurfer 7) | Stage 3: eTIV, nWBV, ASF, hippocampal volume / ICV |
| UCBERKELEY_AMY_6MM, UCBERKELEYFDG_8mm | Stage 4: amyloid Centiloid, FDG-PET MetaROI metabolism |
| DXSUM | 24-month progression target |
| MOCA, CDR | contextual only, never model inputs (CDR excluded to avoid leakage) |

**Not used:**
- **Tau PET (UCBERKELEY_TAU_6MM).** Only 5 converters had a tau scan within 180 days
  of baseline, too few to learn from (tau PET in ADNI starts around 2015).
- **Tracer-specific amyloid SUVR.** It isn't comparable across florbetapir,
  florbetaben and NAV4694; Centiloid is.

## Cohorts

Target: cognitively normal or MCI at baseline with ≥ 24 months of follow-up;
positive = dementia diagnosis within 24 months. Every later-stage measurement is the
one closest to the baseline diagnosis date, within ±180 days.

| | Stage 1 Screening | Stage 2 Blood | Stage 3 MRI | Stage 4 PET |
|---|---|---|---|---|
| Entry requirement | core screening data | any plasma biomarker | MRI | amyloid or FDG PET |
| Participants | 1,677 | 1,071 | 1,645 | 1,365 |
| Converters | 128 (7.6%) | 52 (4.9%) | 128 (7.8%) | 71 (5.2%) |
| Test-split converters | 26 | 11 | 25 | 14 |

## Data handling for real ADNI tables

1. **ADNI missing codes.** `-1`, `-4` and `-5` ("insufficient sample") become
   missing, as do implausible values (e.g. 30 FreeSurfer ICVs outside 900–2300 cm³).
   Negative Centiloids are genuine values and are kept.
2. **Age** is derived from date of birth and the baseline diagnosis date, because
   PTDEMOG has no age column.
3. **MRI measures** are derived from FreeSurfer 7, using codes from the ADNI data
   dictionary. Derived means match OASIS norms (eTIV ≈ 1,488 cm³, nWBV ≈ 0.72).
   - eTIV = ICV `ST10CV`
   - nWBV = (total GM `ST154SV` + cerebral WM `ST152SV` + cerebellar WM + brainstem) / ICV
   - ASF = 1755 / eTIV
   - hippocampus_icv = (`ST29SV` + `ST88SV`) / ICV
4. **BMI** converts pounds and inches using the VITALS unit codes. When the height
   unit was not recorded, it is inferred from the value.
5. **APOE ε4** is the number of ε4 alleles in `GENOTYPE`. Genotype alone does not
   place a patient in Stage 2; a measured blood biomarker does.
6. **The two NfL assays are kept separate**: UPENN Quanterix (mean ≈ 15 pg/mL) and
   the older Blennow-lab assay (mean ≈ 44 pg/mL) are on different scales.
7. **Missingness indicators are off.** The few records missing ADAS/FAQ contained no
   converters, so an indicator learned "missing → low risk" and dropped incomplete
   records to LOW priority. Cross-validated ROC-AUC was unchanged without it.

## Model selection

Logistic regression, random forest and XGBoost were compared on a 60/20/20
participant split. Selection used PR-AUC, top-20% recall, calibration and
interpretability.

| Stage | Selected | Validation ROC-AUC | Validation PR-AUC |
|---|---|---|---|
| 1 Screening | Logistic regression | 0.899 | 0.494 |
| 2 Blood | Random forest | 0.921 | 0.266 |
| 3 MRI | Random forest | 0.914 | 0.479 |
| 4 PET | Logistic regression | 0.918 | 0.339 |

## Untouched test split

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| ROC-AUC | 0.898 | 0.908 | 0.901 | 0.895 |
| PR-AUC | 0.434 | 0.495 | 0.403 | 0.232 |
| Converters in top 20% | 81% | 82% | 80% | 79% |
| Brier | 0.058 | 0.038 | 0.058 | 0.048 |
| ECE (calibration error) | 0.035 | 0.027 | 0.027 | 0.028 |

Figures are from the models trained with the pinned Python 3.15 stack
(`requirements.txt`). Random-forest stages differ in the third decimal from runs on
older scikit-learn versions.

At low prevalence the system prioritises by rank: the HIGH tier (top 20%) captures
about 80% of 24-month converters, against 20% for random selection (≈ 4× lift).

## Comparison with a simple impairment score

Baseline: (30 − MMSE) + ADAS-Cog 11 + FAQ, with no fitting. Each stage's own test split.

| Stage | Model ROC / top-20% | Simple sum ROC / top-20% | Verdict |
|---|---|---|---|
| 1 Screening | 0.898 / 81% | 0.887 / 77% | model slightly better |
| 2 Blood | 0.908 / 82% | 0.928 / 91% | **simple sum better** (11 test converters) |
| 3 MRI | 0.901 / 80% | 0.873 / 64% | model better |
| 4 PET | 0.895 / 79% | 0.838 / 57% | **model clearly better** |

## What each data source adds (repeated 5×5-fold cross-validation)

Same participants, screening features with and without the new modality.

| Stage cohort | Features | LR PR-AUC | RF PR-AUC |
|---|---|---|---|
| Stage 1 (n = 1,677) | screening | 0.459 | 0.411 |
| | + comorbidities | 0.474 | 0.473 |
| Stage 2 (n = 1,071) | screening | 0.388 | 0.372 |
| | + APOE ε4 | 0.425 | 0.398 |
| | + APOE ε4 + plasma | 0.419 | 0.444 |
| Stage 3 (n = 1,645) | screening + age | 0.464 | – |
| | + eTIV / nWBV / ASF + hippocampus | 0.511 | 0.470 |
| Stage 4 (n = 1,365) | screening | 0.363 | 0.336 |
| | + amyloid | 0.363 | 0.386 |
| | + FDG | 0.411 | 0.346 |
| | + amyloid + FDG + APOE ε4 | 0.416 | 0.395 |

ROC-AUC sits between 0.90 and 0.93 throughout. Standard deviations are about
0.07–0.11 for PR-AUC, so most single gains are within about one SD.

## Interpretation

- **Screening tests carry most of the signal.** MMSE, ADAS-Cog and FAQ alone reach
  ROC-AUC ≈ 0.91. Later stages add modest, not dramatic, improvements.
- **The later, more expensive stages add the most where they are used.** The PET and
  MRI models beat the simple score sum on their cohorts, especially at the top of the
  ranking (Stage 4: 79% vs 57% of converters in the top 20%). FDG metabolism and
  hippocampal volume are the most useful additions.
- **Blood biomarkers give limited added value in this cohort.** APOE ε4 helps a little;
  plasma p-tau/NfL/GFAP add little on top. Most blood samples near baseline come from
  ADNI1/GO/2 (p-tau181 and older NfL assay), and the modern p-tau217 panel covers only
  486 participants with 19 converters. On the Stage-2 test split the simple sum did
  better. Report this as a limitation, not a failure.
- **Amyloid PET alone adds little for 24-month timing.** It shows who has
  Alzheimer's pathology rather than who will convert soon. FDG (neuronal dysfunction)
  is more informative for near-term decline.
- **Comorbidities add a small gain** and match the clinical workflow on the
  architecture slide.
- **Small test sets.** Stages 2 and 4 have 11 and 14 test converters, so prefer the
  cross-validated figures.
- **Calibration is good at every stage** (ECE ≈ 0.03).

## TEMPO: how fast will this person decline?

**Target:** annual change in CDR Sum of Boxes (CDR-SB, 0–18), the slope of a
straight line through each participant's CDR-SB visits from baseline to 36 months
(visits up to 39 months allowed). Cohort: the same CN/MCI baseline participants as
the risk models, with a CDR-SB visit at baseline (±3 months), at least 3 visits, and
at least 12 months between the first and last visit. CDR-SB is only the outcome; it
is never a model input.

**Speed tiers** (CDR-SB points per year): STABLE < 0.3, MODERATE 0.3–1.0,
FAST ≥ 1.0. In the Stage-1 cohort that is 70% / 18% / 12%. Participants who
converted to dementia within 24 months declined by 1.56 points/year on average,
against 0.16 for non-converters.

**Method:**
- Ridge regression, random forest and XGBoost are compared per stage on a 60/20/20
  split stratified by speed tier.
- Selection weights: Spearman rank correlation (0.4), fast-decliner capture (0.4)
  and interpretability (0.2). Ridge regression won at every stage.
- The 80% prediction interval is split-conformal: the 80th percentile of 5-fold
  cross-validated absolute residuals.

### Untouched test split

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| Participants (fast decliners) | 1,287 (148) | 804 (76) | 1,272 (146) | 1,021 (97) |
| MAE, points/year | 0.39 | 0.35 | 0.31 | 0.32 |
| MAE if always predicting the mean | 0.47 | 0.42 | 0.44 | 0.40 |
| R² | 0.28 | 0.32 | 0.47 | 0.39 |
| Spearman ρ | 0.36 | 0.38 | 0.49 | 0.48 |
| Fast decliners captured in top 20% | 76% | 80% | 90% | 90% |
| ...by the simple impairment sum | 72% | 80% | 90% | 84% |
| 80% interval half-width | ±0.53 | ±0.52 | ±0.57 | ±0.53 |
| Observed interval coverage | 79% | 78% | 85% | 85% |

### What each data source adds (repeated 5×5-fold CV, ridge regression)

| Stage cohort | Screening only R² / ρ | Stage features R² / ρ |
|---|---|---|
| Stage 1 (comorbidities) | 0.29 / 0.46 | 0.29 / 0.46 |
| Stage 2 (blood + APOE ε4) | 0.26 / 0.40 | 0.27 / 0.40 |
| Stage 3 (MRI) | 0.29 / 0.46 | 0.32 / 0.48 |
| Stage 4 (PET) | 0.23 / 0.41 | 0.29 / 0.45 |

### Interpretation

- **TEMPO finds most fast decliners.** The top 20% by predicted rate contains
  76–90% of people who actually declined by at least 1 point/year. This is the
  property that matters for trial enrichment and follow-up scheduling.
- **Exact rates are only moderately predictable** (R² ≈ 0.3). Most people are
  stable, and individual trajectories are noisy, so the ±0.5 points/year interval is
  shown with every estimate.
- **Imaging adds the most.** PET (+0.06 R²) and MRI (+0.03) improve rate prediction;
  blood biomarkers and comorbidities add essentially nothing, the same pattern as for
  risk.
- **The single Stage-3 test split (R² 0.47) was favourable.** Cross-validation
  suggests about 0.32; prefer the CV figures.
- **The simple impairment sum is again a strong baseline** at Stages 1–3. TEMPO's
  added value is a calibrated rate with an interval, explanations, and the gain at
  Stage 4 (90% vs 84% fast decliners captured).

### Combined outlook

The dashboard pairs each patient's priority tier (who should be tested first) with
their TEMPO tier (how fast they will decline):

| | Slower decline | Fast decline |
|---|---|---|
| **High risk** | Test soon, routine follow-up | **Test first, refer urgently** |
| **Lower risk** | Routine pathway | Re-check sooner |

These are workflow suggestions for clinician review, not diagnoses or treatment
recommendations.

## Reproduce

```bash
for s in stage1 stage2 stage3 stage4; do
  python scripts/02_build_dataset.py --stage $s
  python scripts/03_train.py --stage $s
done
python scripts/10_build_tempo_target.py
for s in stage1 stage2 stage3 stage4; do
  python scripts/11_train_tempo.py --stage $s
done
python scripts/05_evaluate_prioritization.py
python scripts/08_uncertainty_analysis.py --stage stage1 --replicates 500
python scripts/09_sensitivity_analysis.py
python scripts/12_inference_analysis.py              # docs/INFERENCES.md figures
```
