from typing import List, Optional

from pydantic import BaseModel, Field


class PatientAssessment(BaseModel):
    """
    Structured patient assessment across the progressive pathway.

    Stage 1 (cognitive screening):  education, sex, mmse, adas, faq
    Stage 2 (blood biomarkers):     ptau217, ab42_ab40, nfl, gfap
    Stage 3 (MRI):                  eTIV, nWBV, ASF
    Stage 4 (PET):                  amyloid_centiloid, amyloid_suvr, tau_suvr

    Contextual only (never model inputs): moca, cdr.
    CDR is excluded from modelling to avoid label leakage; it is only used to
    route single CDR = 0 patients to the early-detection task.
    """

    assessment_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    patient_ref: str = Field(default="anonymous", min_length=1, max_length=100)

    # Stage 1
    education: Optional[float] = Field(default=None, ge=0, le=30, description="Years of education.")
    sex: Optional[float] = Field(default=None, description="ADNI coding: 1=Male, 2=Female.")
    mmse: Optional[float] = Field(default=None, ge=0, le=30, description="MMSE total score.")
    adas: Optional[float] = Field(default=None, ge=0, le=70, description="ADAS-Cog 11 total score.")
    faq: Optional[float] = Field(default=None, ge=0, le=30, description="Functional Activities Questionnaire total.")

    # Stage 2
    ptau217: Optional[float] = Field(default=None, ge=0, description="Plasma p-tau217 (pg/mL).")
    ab42_ab40: Optional[float] = Field(default=None, ge=0, le=1, description="Plasma Aβ42/Aβ40 ratio.")
    nfl: Optional[float] = Field(default=None, ge=0, description="Plasma neurofilament light (pg/mL).")
    gfap: Optional[float] = Field(default=None, ge=0, description="Plasma GFAP (pg/mL).")

    # Stage 3
    eTIV: Optional[float] = Field(default=None, ge=0, description="Estimated total intracranial volume (cm³).")
    nWBV: Optional[float] = Field(default=None, ge=0, le=1, description="Normalized whole-brain volume.")
    ASF: Optional[float] = Field(default=None, ge=0, description="Atlas scaling factor.")

    # Stage 4
    amyloid_centiloid: Optional[float] = Field(default=None, ge=-100, le=300, description="Amyloid PET Centiloid.")
    amyloid_suvr: Optional[float] = Field(default=None, ge=0, description="Amyloid PET summary SUVR.")
    tau_suvr: Optional[float] = Field(default=None, ge=0, description="Tau PET meta-temporal SUVR.")

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
