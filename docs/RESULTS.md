# Results on ADNI (October 2026 download)

Aggregate results only. No subject-level ADNI data is stored in this repository.

Data used in preparation of this work were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu).

## Data available

| Table | Used for |
|---|---|
| PTDEMOG, MMSE, ADAS, FAQ, MOCA | Stage 1 inputs (MoCA contextual only) |
| DXSUM | 24-month progression target |
| UCSFFSX7 (FreeSurfer 7) | Stage 3 MRI features |
| CDR | not a model input (leakage control) |
| All_Subjects_Key_MRI | scan metadata only; holds no volumes |

**No plasma-biomarker or PET tables were in the download**, so Stage 2
(blood biomarkers) and Stage 4 (PET) are implemented and tested but **not trained**.
To enable them, add `UPENN_PLASMA_FUJIREBIO_QUANTERIX_*.csv` and
`UCBERKELEY_AMY_6MM_*.csv` / `UCBERKELEY_TAU_6MM_*.csv` to `data/raw/` and run the
build/train commands for `stage2` / `stage4`.

## Cohort

Target: CN or MCI at baseline with ≥ 24 months of follow-up; positive = dementia
diagnosis within 24 months.

| | Stage 1 | Stage 3 |
|---|---|---|
| Participants | 1,677 | 1,645 |
| Converters (24 m) | 128 (7.63%) | 128 (7.78%) |
| Records with all inputs | 98.7% | 95.8% |
| Train / validation / test | 1,006 / 335 / 336 | 987 / 329 / 329 |

## Data handling changes made for real ADNI tables

1. **ADNI missing codes.** Negative sentinel codes (`-1`, `-4`) were previously
   treated as real values (e.g. three FAQ scores of `-1`). They are now converted to
   missing, along with implausible values (e.g. 30 FreeSurfer ICVs outside
   900–2300 cm³).
2. **Age.** ADNI PTDEMOG has no `AGE` column, so age was "100% missing". It is now
   derived from date of birth (`PTDOB`) and the baseline diagnosis date.
3. **MRI features.** ADNI has no OASIS `eTIV`/`nWBV`/`ASF` columns, so the original
   Stage-3 configuration could not run. They are derived from FreeSurfer 7
   (`UCSFFSX7`), using codes from the ADNI data dictionary. Derived means match OASIS
   norms (eTIV ≈ 1,488 cm³, nWBV ≈ 0.72):
   - eTIV = ICV (`ST10CV`)
   - nWBV = (total GM `ST154SV` + cerebral WM `ST152SV` + cerebellar WM + brainstem) / ICV
   - ASF = 1755 / eTIV
   - hippocampus_icv = (`ST29SV` + `ST88SV`) / ICV
4. **Scan timing.** Each participant's MRI is the scan closest to the baseline
   diagnosis date within ±180 days. Selecting by visit code alone matched only 1,006
   participants and could pick scans years after baseline. This rule matches 1,645.
5. **Missingness indicators turned off.** The 22 participants missing ADAS or FAQ
   included no converters. An indicator feature therefore learned "missing → low
   risk" and dropped incomplete records to the LOW tier (risk 0.2%). Without it, risk
   stays at a neutral imputed value (MEDIUM). Cross-validated ROC-AUC was unchanged
   (0.9145 vs 0.9137).

## Model selection (validation split)

| Stage | Model | ROC-AUC | PR-AUC | Top-20% recall | ECE | Selection score |
|---|---|---|---|---|---|---|
| 1 | **Logistic regression** | 0.897 | 0.502 | 0.76 | 0.028 | **0.737** |
| 1 | Random forest | 0.885 | 0.446 | 0.68 | 0.020 | 0.661 |
| 1 | XGBoost | 0.873 | 0.367 | 0.72 | 0.030 | 0.631 |
| 3 | **Random forest** | 0.908 | 0.495 | 0.81 | 0.020 | **0.723** |
| 3 | Logistic regression | 0.903 | 0.335 | 0.77 | 0.035 | 0.681 |
| 3 | XGBoost | 0.877 | 0.412 | 0.65 | 0.010 | 0.627 |

## Untouched test split

| | Stage 1 (logistic regression) | Stage 3 (random forest) |
|---|---|---|
| ROC-AUC | 0.894 | 0.881 |
| PR-AUC (prevalence ≈ 0.077) | 0.391 | 0.393 |
| Top-20% recall | 0.769 | 0.720 |
| Brier | 0.061 | 0.060 |
| ECE | 0.023 | 0.024 |
| Recall / specificity at p ≥ 0.5 | 0.115 / 0.984 | 0.160 / 0.984 |

The 0.5 cut-off is reported for completeness only. At 7.6% prevalence the system
prioritises by rank: the top 20% (HIGH tier) captures about three quarters of 24-month
converters, against 20% for random selection (≈ 3.8× lift).

## Robustness check: repeated 5×5-fold cross-validation (Stage-3 cohort, n = 1,645)

| Features | Model | ROC-AUC (SD) | PR-AUC (SD) |
|---|---|---|---|
| Stage 1 without age | LR | 0.916 (0.021) | 0.469 (0.079) |
| Stage 1 + age | LR | 0.915 (0.023) | 0.464 (0.083) |
| + eTIV / nWBV / ASF | LR | 0.913 (0.021) | 0.486 (0.075) |
| + hippocampus_icv | LR | 0.920 (0.017) | 0.511 (0.068) |
| + hippocampus_icv | RF | 0.916 (0.016) | 0.470 (0.066) |

## Comparison with a simple impairment score

Baseline: (30 − MMSE) + ADAS-Cog 11 + FAQ, with no fitting. Same test split:

| | ROC-AUC | PR-AUC | Top-20% recall |
|---|---|---|---|
| Stage 1 model | 0.894 | 0.391 | 0.769 |
| Simple impairment sum | 0.887 | 0.388 | 0.769 |
| Stage 3 model (with MRI) | 0.881 | 0.393 | 0.720 |
| Simple impairment sum (Stage-3 test split) | 0.873 | 0.337 | 0.640 |

## Interpretation

- **Stage 1 is close to a simple impairment sum.** On ADNI, the calibrated model ranks
  patients about as well as adding up the three test scores. Its value is calibrated
  risk, explanations and integration with the pathway, not better ranking. This should
  be stated plainly in the report.
- **Age adds nothing measurable** once cognitive and functional scores are known.
- **The OASIS-style MRI measures (eTIV, nWBV, ASF) add little. Hippocampal volume adds
  a modest gain** (CV PR-AUC 0.464 → 0.511, about one SD). On the test split, the
  Stage-3 model beats the simple sum in top-20% recall (0.72 vs 0.64).
- **The test set is small** (~25 converters), so single-split differences of a few
  points are within noise. Prefer the cross-validated figures.
- **Calibration is good** (ECE ≈ 0.02), which supports showing risk scores as
  probabilities for prioritisation research.

## Reproduce

```bash
for s in stage1 stage3; do
  python scripts/02_build_dataset.py --stage $s
  python scripts/03_train.py --stage $s
done
python scripts/05_evaluate_prioritization.py
python scripts/08_uncertainty_analysis.py --stage stage1 --replicates 500
python scripts/09_sensitivity_analysis.py
```
