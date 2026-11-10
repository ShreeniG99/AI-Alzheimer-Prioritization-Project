# What to infer from the results

Answers to the review questions, with the figures behind them. Aggregates only; no
participant-level ADNI data is stored here.

Figures come from `scripts/12_inference_analysis.py` (output:
`reports/inference_summary.json`) run on the October 2026 ADNI download with the
models trained on the pinned Python 3.15 stack, and from `docs/RESULTS.md`.
"Cross-validated" means repeated 5×5-fold logistic regression with the project's
preprocessing; "out-of-fold" means every participant is scored by a model that did
not see them.

## Headline inferences

1. **Three cheap tests carry the prediction.** MMSE + ADAS-Cog + FAQ alone reach
   ROC-AUC 0.915, the same as all 14 Stage-1 inputs.
2. **Later tests sharpen the top of the queue, not the overall AUC.** On the same
   participants PET moves ROC-AUC 0.918 → 0.928 and top-20% capture 87% → 91%.
3. **The misses are converters who still look healthy.** 64% of the converters the
   Stage-1 ranking missed had MMSE ≥ 28 at baseline, against 28% of those it caught.
4. **The pathway saves tests.** The LOW tier was 100 of 336 test participants and
   contained no converter.
5. **Performance is uneven.** ROC-AUC is 0.82 for 12 years of education or less,
   against 0.91–0.92 above. Validate locally before any use.

## Q1. ROC-AUC is about 0.90. Who are the other 10%?

Not 10% of patients. ROC-AUC counts pairs: the Stage-1 test split has 26 converters
× 310 non-converters = 8,060 pairs, and 821 (10.2%) are ranked the wrong way round.

Converters missed vs caught by the top 20% (out-of-fold, all 1,677 participants):

| | Missed (25) | Caught (103) |
|---|---|---|
| MMSE ≥ 28 | 64% | 28% |
| FAQ = 0 | 32% | 0% |
| Median ADAS-Cog | 9.3 | 14 |
| Median MMSE / FAQ | 28 / 2 | 26 / 7 |

ROC-AUC inside each subgroup (out-of-fold; overall 0.908):

| Group | Participants | Converters | ROC-AUC | Top-20% capture within group |
|---|---|---|---|---|
| Female | 822 | 49 | 0.912 | 86% |
| Male | 855 | 79 | 0.902 | 78% |
| Age under 70 | 611 | 34 | 0.904 | 85% |
| Age 70–79 | 812 | 68 | 0.911 | 84% |
| Age 80 and over | 254 | 26 | 0.893 | 69% |
| No smoking history | 690 | 74 | 0.904 | 76% |
| Smoking history | 458 | 49 | 0.884 | 73% |
| Education over 16 years | 741 | 44 | 0.914 | 84% |
| Education 13–16 years | 741 | 61 | 0.919 | 82% |
| Education 12 years or less | 195 | 23 | 0.823 | 57% |

Errors follow disease stage, not sex or smoking. The low-education group is small
(23 converters), so its figure is uncertain.

## Q2 and Q5. Why do stage AUCs differ? Why not start at Stage 2?

Each stage is tested on different participants, so the test figures cannot be ranked:

| Stage | Test participants (converters) | ROC-AUC | Bootstrap 95% interval |
|---|---|---|---|
| 1 | 336 (26) | 0.898 | 0.832–0.949 |
| 2 | 215 (11) | 0.908 | 0.829–0.970 |
| 3 | 329 (25) | 0.901 | 0.856–0.942 |
| 4 | 273 (14) | 0.895 | 0.845–0.941 |

Same participants, screening inputs only vs plus that stage's tests (cross-validated):

| Cohort | ROC-AUC | PR-AUC | Top-20% capture |
|---|---|---|---|
| Stage 2 (1,071; 52 converters) | 0.929 → 0.931 | 0.388 → 0.416 | 88% → 90% |
| Stage 3 (1,645; 128 converters) | 0.915 → 0.925 | 0.481 → 0.512 | 82% → 84% |
| Stage 4 (1,365; 71 converters) | 0.918 → 0.928 | 0.375 → 0.416 | 87% → 91% |

Blood adds almost nothing on top of screening. Stage 2 looked better because its
participants were easier to rank (screening alone gives 0.929 there, 0.915 in the
Stage-1 cohort). The pathway is still needed: every later model uses the Stage-1
inputs, every later cohort is a subset of Stage 1, and the LOW tier (30% of the test
split, no converters) could be spared the later tests.

## Q3. Why is the PET stage only 0.895?

0.895 is a test figure on a different, smaller cohort (14 test converters). On the
same participants PET improves every measure (table above), most at the top of the
ranking; on the Stage-4 test split the model finds 79% of converters in the top 20%
against 57% for the simple score sum. The gain is modest: ROC-AUC rises 0.010 while
the fold-to-fold standard deviation is about 0.025. FDG carries the gain; amyloid
alone adds nothing for 24-month timing (`docs/RESULTS.md`).

## Q4. Are the same participants in all four stages?

Yes, checked on participant IDs. Stages 2, 3 and 4 (1,071 / 1,645 / 1,365) are all
subsets of Stage 1 (1,677). 987 participants are in all four stages, 44 of them
converters. The outcome is built once per participant from DXSUM, so no participant
has conflicting labels across stages.

Caveat: each stage is split 60/20/20 separately, so a participant can be in one
stage's training set and another stage's test set. Nothing leaks within a stage; a
single shared split would be cleaner.

## Q6. Why is education a feature?

It is years of formal education. More education goes with lower risk, and it lets
the model read test scores fairly:

| Education | Participants | Converted | Mean MMSE of stable participants |
|---|---|---|---|
| 12 years or less | 195 | 11.8% | 27.85 |
| 13–16 years | 741 | 8.2% | 28.38 |
| Over 16 years | 741 | 5.9% | 28.75 |

Its standardised coefficient in the Stage-1 model is −0.08 (ADAS-Cog +1.16, FAQ
+1.09, MMSE −0.48). These are associations in ADNI, not causal effects.

## Q7. Can staff enter fewer inputs?

Yes (Stage-1 cohort, cross-validated):

| Inputs | ROC-AUC | PR-AUC | Top-20% capture |
|---|---|---|---|
| All 14 Stage-1 inputs | 0.915 | 0.474 | 82% |
| Age, sex, education, MMSE, ADAS-Cog, FAQ | 0.914 | 0.459 | 82% |
| MMSE, ADAS-Cog, FAQ | 0.915 | 0.465 | 81% |
| ADAS-Cog, FAQ | 0.912 | 0.464 | 80% |
| FAQ only | 0.879 | 0.380 | 76% |

The eight comorbidity inputs are optional in the app and are imputed when blank.
The data-quality grade is computed on the six core inputs, so those should still be
entered.

## Q8. Why trust the score?

- Evaluated once on a 20% split never used for training, model choice or calibration.
- Calibrated: ECE 0.027–0.035.
- Out-of-range inputs are rejected by the API schema (e.g. MMSE 0–30, ADAS-Cog 0–70,
  FAQ 0–30, age 40–110).
- One-at-a-time sensitivity (`scripts/09_sensitivity_analysis.py`): FAQ 0 → 30 moves
  the risk 3.2% → 93.4%; ADAS-Cog 0 → 40, 0.6% → 81.6%; MMSE 30 → 18, 2.3% → 15.8%;
  age 60 → 90, 3.2% → 3.3%.
- Every score carries SHAP reasons and its model version; every action is audit-logged.

Limits: a plausible typo cannot be detected; bootstrap intervals for the risk exist
only for cohort participants, not for a new patient; nothing is validated outside ADNI.

## Q9. What does "the top 20% captures 81%" mean?

Stage-1 test split: 336 participants, 26 converters.

| | Participants | Converters |
|---|---|---|
| Top 20% by risk | 68 | 21 (81%) |
| Remaining 80% | 268 | 5 (19%) |
| Random 68, for comparison | 68 | about 5 |

By tier (cut-points fixed on development data): HIGH 75 participants / 21 converters,
MEDIUM 161 / 5, LOW 100 / 0. The five converters outside the top 20% are all in the
MEDIUM tier (clinician review; re-ranked when new results arrive). 47 of the 68
tested first did not convert within 24 months.

At a 0.5 probability cut-off the same model flags only 3 of the 26 converters
(recall 12%). The system ranks; it does not classify at a cut-off.
