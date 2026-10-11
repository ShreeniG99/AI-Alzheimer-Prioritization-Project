# AI-Driven Prioritization for Early Alzheimer's Diagnostic Pathways

Clinical decision-support **research prototype** that ranks patients for further
Alzheimer's diagnostic evaluation under limited capacity (blood tests, MRI, PET,
clinician time). It estimates a calibrated 24-month progression risk, assigns a
LOW / MEDIUM / HIGH priority tier, explains every score with SHAP, and routes
patients along a progressive diagnostic pathway.

> It does **not** diagnose Alzheimer's disease and does **not** recommend
> treatment. Every output supports clinician-led decisions.

Team: Harithra S, Tharika N S, Thanushika Sri RA, Shreenidhi G (Amrita Vishwa Vidyapeetham).

---

## What the system does

```
Clinician signs in (JWT, audit-logged)
        │
        ▼
Stage 1  Cognitive screening ──► risk + tier ──► HIGH? escalate
Stage 2  Blood biomarkers    ──► refined risk ──► HIGH? escalate
Stage 3  MRI evaluation      ──► refined risk ──► HIGH? escalate
Stage 4  PET imaging         ──► refined risk ──► specialist review
        │
        ├─ SHAP explanation of each score
        ├─ score, tier, reasoning and pathway trace saved to PostgreSQL
        └─ capacity-aware triage queue, re-ranked whenever new data arrives
```

| Stage | Model inputs | ADNI source tables (prefix) |
|---|---|---|
| 1 Cognitive screening | age, education, sex, MMSE, ADAS-Cog 11, FAQ + comorbidities (BMI, systolic BP, GDS depression, cardiovascular / endocrine / psychiatric / neurologic / smoking history) | `PTDEMOG_`, `MMSE_`, `ADAS_`, `FAQ_`, `VITALS_`, `GDSCALE_`, `MEDHIST_` |
| 2 Blood biomarkers | Stage 1 + APOE ε4, plasma p-tau217, p-tau181, Aβ42/40, NfL, GFAP | `UPENN_PLASMA_FUJIREBIO_QUANTERIX_`, `UGOTPTAU181_`, `ADNI_BLENNOWPLASMANFLLONG_`, `APOERES_` |
| 3 MRI | Stage 1 + eTIV, nWBV, ASF, hippocampal volume/ICV (+ blood results when available) | `UCSFFSX7_` (FreeSurfer 7; measures derived, see below) |
| 4 PET | Stage 1 + amyloid Centiloid, FDG-PET metabolism (+ earlier modalities) | `UCBERKELEY_AMY_6MM_`, `UCBERKELEYFDG_` |

**Target** (`config.yaml → target.mode: progression_24m`): among participants who are
cognitively normal or MCI at baseline with ≥ 24 months of follow-up, `1` = dementia
diagnosis (DXSUM) within 24 months, `0` otherwise.

**Results on ADNI:** see [`docs/RESULTS.md`](docs/RESULTS.md). All four stages are
trained:

| Stage | Participants | Converters | Test ROC-AUC | Converters in top 20% |
|---|---|---|---|---|
| 1 Screening | 1,677 | 128 | 0.898 | 81% |
| 2 Blood | 1,071 | 52 | 0.903 | 82% |
| 3 MRI | 1,645 | 128 | 0.900 | 80% |
| 4 PET | 1,365 | 71 | 0.895 | 79% |

Tau PET is not used: only 5 converters had a tau scan near baseline.

**ADNI-specific handling:**
- Negative missing codes (`-1`, `-4`, `-5`) become missing values; negative Centiloids are kept.
- Age is derived from `PTDOB`.
- eTIV/nWBV/ASF/hippocampal volume are computed from FreeSurfer volumes.
- Each later-stage measurement is the one closest to the baseline date (±180 days).
- Missingness indicators are off, so incomplete records are not pushed to LOW.

**Leakage control:** CDR is never a model input. It is only an optional routing field
(a single patient with CDR = 0 is handled as an *early-detection* task). MoCA is
contextual only (69% missing in the cohort).

### Task routing (architecture slide)

| Task | Endpoint | Description |
|---|---|---|
| Single assessment / early detection | `POST /assessment` | Score along the pathway, explain, save to queue. `cdr = 0` → `early_detection`. |
| Triage | `POST /triage` | Rank the saved queue against available capacity (optionally one stage). |
| New-data re-rank | `PATCH /patients/{id}` | Add biomarker / MRI / PET results → re-score → re-rank the queue. |

### Models

For each stage, logistic regression, random forest and XGBoost are compared on a
participant-level 60/20/20 split. The selection score weights PR-AUC (0.35), top-k
recall (0.35), calibration (0.15) and interpretability (0.15). The winner is refit
on train+validation, Platt-calibrated on out-of-fold predictions, and evaluated once
on the untouched test split (ROC-AUC, PR-AUC, recall, precision, specificity, Brier,
ECE, top-20% recall).

Priority tiers are **capacity cut-points**, not clinical cutoffs. The top 20% of
development-set risk is HIGH, the next 50% is MEDIUM, and the rest is LOW
(`config.yaml → priority`).

Explanations use SHAP (`LinearExplainer` / `TreeExplainer`) on the preprocessed
features. Contributions from imputation/missingness indicators and one-hot columns
are aggregated back to the original clinical inputs.

---

## Repository layout

```
config.yaml                 features per stage, target, split, selection weights
scripts/
  00_make_synthetic_data.py SYNTHETIC ADNI-shaped tables (tests/demo only)
  01_profile_data.py        profile raw tables before modelling
  02_build_dataset.py       build data/processed/<stage>.csv
  03_train.py               compare models, calibrate, save artifacts/<stage>_model.joblib
  04_sanity_check.py        artifact/dataset checks
  05_evaluate_prioritization.py  capacity curves vs random/clinical baselines
  06_data_quality_analysis.py    missingness analysis
  07_feature_distribution_analysis.py
  08_uncertainty_analysis.py     bootstrap refits -> <stage>_bootstrap_predictions.npz
  09_sensitivity_analysis.py     one-at-a-time input sensitivity
ml/                         preprocessing, models, calibration, metrics, SHAP
backend/                    FastAPI app, auth, audit log, pathway + model registry
frontend/                   React (Vite) clinician dashboard
tests/                      pytest: full pipeline on synthetic data + API
deploy/aws-ec2.md           single-EC2 Docker deployment
```

---

## 1. Train on ADNI data

ADNI data is access-controlled. **Never commit or share subject-level data**,
including `data/processed/*.csv` and the trained `artifacts/` (model bundles hold a
small background sample of training rows for SHAP). All of these are git-ignored.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the unzipped ADNI CSVs into `data/raw/`. The scripts pick the newest file per
prefix, so there's no need to rename anything. Stages 2–4 are optional: a stage whose
tables are missing is simply not trained, and the API runs with the stages that exist.

```bash
python scripts/01_profile_data.py                       # review before modelling

for s in stage1 stage2 stage3 stage4; do
  python scripts/02_build_dataset.py --stage $s
  python scripts/03_train.py --stage $s
  python scripts/04_sanity_check.py --stage $s
done

python scripts/05_evaluate_prioritization.py            # reports/prioritized_cohort.csv etc.
python scripts/06_data_quality_analysis.py
python scripts/07_feature_distribution_analysis.py
python scripts/08_uncertainty_analysis.py --stage stage1 --replicates 500
python scripts/09_sensitivity_analysis.py
```

If ADNI column names differ from the ones expected, add them to `ALIASES` in
`scripts/02_build_dataset.py`.

## 2. Try it without ADNI access (synthetic demo)

`scripts/00_make_synthetic_data.py` writes **artificial** tables with ADNI file
names and columns. They are for demos and automated tests only. Nothing trained on
them says anything about real patients.

```bash
python scripts/00_make_synthetic_data.py --out data/raw_synthetic
export ALZ_RAW_DIR=data/raw_synthetic       # PowerShell: $env:ALZ_RAW_DIR="data/raw_synthetic"
# then run the same 02/03/05/08 commands as above
```

`ALZ_RAW_DIR`, `ALZ_PROCESSED_DIR`, `ALZ_ARTIFACTS_DIR` and `ALZ_REPORTS_DIR` override
the folders in `config.yaml` for both the scripts and the backend.

## 3. Run the application locally

```bash
export JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
export ADMIN_USERNAME=admin ADMIN_PASSWORD='choose-a-strong-password'
uvicorn backend.main:app --reload --port 8000           # SQLite app.db by default
```

```bash
cd frontend && npm install && npm run dev               # http://localhost:5173
```

Sign in with the admin account and create clinician accounts with `POST /auth/users`.
You can also use the CLI: `python -m backend.init_db --create-user dr.smith`.
Interactive API docs live at `http://localhost:8000/docs` (use **Authorize**).

| Endpoint | Purpose |
|---|---|
| `POST /auth/login`, `POST /auth/token` | JWT sign-in (JSON / OAuth2 form) |
| `GET /auth/me`, `GET/POST /auth/users` | current user; admin user management |
| `GET /audit` | admin audit log (sign-ins, assessments, views, updates, triage, deletes) |
| `GET /model-info` | deployed stages, models, cohort stats, test metrics, thresholds |
| `POST /assessment` | score + SHAP + save (single / early-detection task) |
| `GET /patients`, `GET/PATCH/DELETE /patients/{id}` | queue, detail, new-data re-rank, removal |
| `POST /triage` | rank saved queue with capacity |
| `POST /predict`, `/explain`, `/quality`, `/uncertainty`, `/rank` | stateless helpers |
| `GET /health` | public liveness + available stages |

## 4. Docker

```bash
cp .env.example .env          # set POSTGRES_PASSWORD, JWT_SECRET, ADMIN_PASSWORD
docker compose up --build
```

This runs three containers: `frontend` (nginx, http://localhost:3000), `backend`
(FastAPI, http://localhost:8000/docs) and `postgres`. `artifacts/`, `reports/` and
`data/processed/` are mounted read-only, never baked into images. For AWS, see
[`deploy/aws-ec2.md`](deploy/aws-ec2.md).

## 5. Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite generates synthetic data in a temporary folder. It builds and trains all
four stages, runs the bootstrap and evaluation scripts, and then exercises the API:
auth, roles, assessment, SHAP, pathway progression, triage, re-ranking, deletion and
validation. GitHub Actions runs it on every push, along with the frontend build and
both Docker image builds.

---

## Scope and limitations

- Research prototype on a retrospective cohort; not a validated medical device.
- The slides list OASIS-2 as the original primary dataset. The implementation uses
  ADNI because it provides the longitudinal diagnoses needed for a prospective
  24-month target, plus plasma and PET tables for Stages 2–4. OASIS-3/4 can be added
  as further sources through the same alias mechanism.
- Priority thresholds are resource-allocation cut-points, not clinical cutoffs.
- SHAP values describe model behaviour, not causal or independent clinical effects.
- Bootstrap intervals are reported only for participants in the retrospective cohort
  and are not clinical confidence intervals.
- For real deployment: TLS, managed secrets, private networking, backups,
  monitoring and institutional governance/clinical validation are required.
