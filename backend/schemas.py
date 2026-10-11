from typing import List, Optional

from pydantic import BaseModel, Field


class PatientAssessment(BaseModel):
    """
    Structured patient assessment across the progressive pathway.

    Stage 1 (cognitive screening):  age, education, sex, mmse, adas, faq,
                                    bmi, sbp, gds, hx_* comorbidity flags
    Stage 2 (blood biomarkers):     ptau217, ab42_ab40, nfl, gfap, ptau181,
                                    nfl_legacy, apoe4
    Stage 3 (MRI):                  eTIV, nWBV, ASF, hippocampus_icv
    Stage 4 (PET):                  amyloid_centiloid, fdg

    Contextual only (never model inputs): moca, cdr.
    CDR is excluded from modelling to avoid label leakage; it is only used to
    route single CDR = 0 patients to the early-detection task.
    """

    assessment_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    patient_ref: str = Field(default="anonymous", min_length=1, max_length=100)

    # Stage 1
    age: Optional[float] = Field(default=None, ge=40, le=110, description="Age in years at assessment.")
    education: Optional[float] = Field(default=None, ge=0, le=30, description="Years of education.")
    sex: Optional[float] = Field(default=None, description="ADNI coding: 1=Male, 2=Female.")
    mmse: Optional[float] = Field(default=None, ge=0, le=30, description="MMSE total score.")
    adas: Optional[float] = Field(default=None, ge=0, le=70, description="ADAS-Cog 11 total score.")
    faq: Optional[float] = Field(default=None, ge=0, le=30, description="Functional Activities Questionnaire total.")

    # Stage 1 comorbidities
    bmi: Optional[float] = Field(default=None, ge=12, le=60, description="Body-mass index (kg/m²).")
    sbp: Optional[float] = Field(default=None, ge=60, le=260, description="Systolic blood pressure (mmHg).")
    gds: Optional[float] = Field(default=None, ge=0, le=15, description="Geriatric Depression Scale total (0-15).")
    hx_cardiovascular: Optional[float] = Field(default=None, ge=0, le=1, description="History of cardiovascular disease (1/0).")
    hx_endocrine: Optional[float] = Field(default=None, ge=0, le=1, description="Endocrine-metabolic disease, e.g. diabetes (1/0).")
    hx_psychiatric: Optional[float] = Field(default=None, ge=0, le=1, description="Psychiatric history (1/0).")
    hx_neurologic: Optional[float] = Field(default=None, ge=0, le=1, description="Neurologic history other than AD (1/0).")
    hx_smoking: Optional[float] = Field(default=None, ge=0, le=1, description="Smoking history (1/0).")

    # Stage 2
    ptau217: Optional[float] = Field(default=None, ge=0, description="Plasma p-tau217 (pg/mL).")
    ab42_ab40: Optional[float] = Field(default=None, ge=0, le=1, description="Plasma Aβ42/Aβ40 ratio.")
    nfl: Optional[float] = Field(default=None, ge=0, description="Plasma neurofilament light (pg/mL).")
    gfap: Optional[float] = Field(default=None, ge=0, description="Plasma GFAP (pg/mL).")
    ptau181: Optional[float] = Field(default=None, ge=0, description="Plasma p-tau181 (pg/mL, Gothenburg Simoa).")
    nfl_legacy: Optional[float] = Field(
        default=None, ge=0, description="Plasma NfL, older Blennow-lab assay (pg/mL; different scale from nfl).",
    )
    apoe4: Optional[float] = Field(default=None, ge=0, le=2, description="Number of APOE ε4 alleles (0, 1, 2).")

    # Stage 3
    eTIV: Optional[float] = Field(default=None, ge=0, description="Estimated total intracranial volume (cm³).")
    nWBV: Optional[float] = Field(default=None, ge=0, le=1, description="Normalized whole-brain volume.")
    ASF: Optional[float] = Field(default=None, ge=0, description="Atlas scaling factor.")
    hippocampus_icv: Optional[float] = Field(
        default=None, ge=0, le=0.02,
        description="Total hippocampal volume / intracranial volume (FreeSurfer).",
    )

    # Stage 4
    amyloid_centiloid: Optional[float] = Field(default=None, ge=-100, le=300, description="Amyloid PET Centiloid.")
    fdg: Optional[float] = Field(default=None, ge=0.5, le=2.0, description="FDG-PET MetaROI mean (glucose metabolism).")

    # Contextual
    moca: Optional[float] = Field(default=None, ge=0, le=30, description="MoCA total. Contextual only.")
    cdr: Optional[float] = Field(default=None, ge=0, le=3, description="CDR global. Routing only, never a model input.")


class PredictionRequest(PatientAssessment):
    stage: Optional[str] = Field(default=None, description="Force a stage model; default is the deepest reached stage.")


class RankingRequest(BaseModel):
    patients: List[PatientAssessment] = Field(min_length=1, description="Patients to rank.")
    capacity: Optional[int] = Field(default=None, ge=1, description="Simulated resource capacity, e.g. MRI slots.")


class TriageRequest(BaseModel):
    capacity: Optional[int] = Field(default=None, ge=1, description="Available diagnostic slots.")
    stage: Optional[str] = Field(default=None, description="Only rank patients currently at this stage.")


class PatientUpdate(BaseModel):
    """New data for a saved patient (e.g. biomarker, MRI or PET results)."""

    values: PatientAssessment
    capacity: Optional[int] = Field(default=None, ge=1)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=100, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(min_length=8, max_length=200)
    full_name: str = Field(default="", max_length=200)
    role: str = Field(default="clinician", pattern="^(clinician|admin)$")
