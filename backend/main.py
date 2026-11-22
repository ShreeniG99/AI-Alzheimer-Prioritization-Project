import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import Base, SessionLocal, engine, get_db
from backend.models import AuditLog, PatientRecord, User
from backend.schemas import (
    LoginRequest,
    PatientAssessment,
    PatientUpdate,
    PredictionRequest,
    RankingRequest,
    TriageRequest,
    UserCreate,
)
from backend.security import (
    audit,
    authenticate,
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
)
from backend.services.model_registry import STAGE_INFO, STAGES, registry
from backend.services.pathway_service import ModelUnavailableError, evaluate_pathway, reached_stages
from backend.services.population_service import get_population_risk_distribution
from backend.services.quality_service import calculate_quality
from backend.services.ranking_service import rank_scored
from backend.services.uncertainty_service import uncertainty_service
from backend.settings import ADMIN_PASSWORD, ADMIN_USERNAME, CORS_ORIGINS, SCOPE

logger = logging.getLogger("alzheimer.api")

ID_FIELDS = {"assessment_id", "patient_ref", "stage"}


def seed_admin() -> None:
    """Create the first administrator from ADMIN_USERNAME / ADMIN_PASSWORD."""
    if not (ADMIN_USERNAME and ADMIN_PASSWORD):
        return
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            db.add(User(
                username=ADMIN_USERNAME,
                full_name="Administrator",
                password_hash=hash_password(ADMIN_PASSWORD),
                role="admin",
            ))
            db.commit()
            logger.info("Created initial administrator '%s'.", ADMIN_USERNAME)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    seed_admin()
    yield


app = FastAPI(
    title="AI Alzheimer Prioritization API",
    description=(
        "Research clinical decision-support API for prioritizing patients along a "
        "progressive diagnostic pathway (cognitive screening -> blood biomarkers -> "
        "MRI -> PET). It does not diagnose or recommend treatment."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in ID_FIELDS}


def run_pathway(features: Dict[str, Any], explain: bool = True) -> Dict[str, Any]:
    try:
        return evaluate_pathway(features, explain=explain)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


def stage_model(stage: Optional[str]):
    if stage is not None and stage not in STAGES:
        raise HTTPException(status_code=422, detail=f"Unknown stage '{stage}'. Use one of {STAGES}.")
    model = registry.get(stage or "stage1")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No {stage or 'stage1'} model is deployed. Train it with scripts/03_train.py.",
        )
    return model


def stage_measurements(stage: str) -> List[str]:
    """The measurements that define a stage (graded for data quality)."""
    info = STAGE_INFO[stage]
    return info.get("trigger_features", info["modality_features"])


def task_for(features: Dict[str, Any]) -> Dict[str, str]:
    if features.get("cdr") == 0:
        return {
            "task": "early_detection",
            "description": "Single patient with CDR = 0: early-detection screening for preclinical risk.",
        }
    return {"task": "single_assessment", "description": "Single patient risk assessment."}


def record_to_dict(record: PatientRecord, detail: bool = False) -> Dict[str, Any]:
    data = {
        "id": record.id,
        "assessment_id": record.assessment_id,
        "patient_ref": record.patient_ref,
        "inputs": record.inputs,
        "current_stage": record.current_stage,
        "current_stage_name": STAGE_INFO.get(record.current_stage, {}).get("name", record.current_stage),
        "risk_score": record.risk_score,
        "priority": record.priority,
        "model_version": record.model_version,
        "next_step": (record.pathway or {}).get("next_step"),
        "decline": decline_summary((record.pathway or {}).get("decline")),
        "created_by": record.created_by,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }
    if detail:
        data["explanation"] = record.explanation
        data["pathway"] = record.pathway
    return data


def decline_summary(decline: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """TEMPO estimate without its (large) SHAP explanation, for lists."""
    if not decline:
        return None
    return {k: v for k, v in decline.items() if k != "explanation"}


def apply_pathway(record: PatientRecord, result: Dict[str, Any]) -> None:
    record.current_stage = result["current_stage"]
    record.risk_score = result["risk_score"]
    record.priority = result["priority"]
    record.model_version = result["model_version"]
    record.explanation = result["explanation"]
    record.pathway = {k: v for k, v in result.items() if k != "explanation"}


def ranked_queue(db: Session, capacity: Optional[int] = None, stage: Optional[str] = None) -> Dict[str, Any]:
    query = db.query(PatientRecord)
    if stage:
        query = query.filter(PatientRecord.current_stage == stage)
    entries = [record_to_dict(r) for r in query.all()]
    try:
        return rank_scored(entries, capacity)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def get_record(db: Session, record_id: int) -> PatientRecord:
    record = db.get(PatientRecord, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient assessment not found.")
    return record


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "available_stages": registry.available(),
        "tempo_stages": registry.tempo_available(),
    }


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _login(db: Session, request: Request, username: str, password: str) -> Dict[str, Any]:
    user = authenticate(db, username, password)
    if user is None:
        audit(db, request, username, "login_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    audit(db, request, user.username, "login")
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "user": {"username": user.username, "full_name": user.full_name, "role": user.role},
    }


@app.post("/auth/login")
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    return _login(db, request, body.username, body.password)


@app.post("/auth/token", include_in_schema=True)
def login_form(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """OAuth2 password flow, used by the interactive /docs 'Authorize' button."""
    return _login(db, request, form.username, form.password)


@app.get("/auth/me")
def me(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {"username": user.username, "full_name": user.full_name, "role": user.role}


@app.get("/auth/users")
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    return [
        {"username": u.username, "full_name": u.full_name, "role": u.role, "is_active": u.is_active}
        for u in db.query(User).order_by(User.username).all()
    ]


@app.post("/auth/users", status_code=201)
def create_user(
    body: UserCreate,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=409, detail="Username already exists.")
    db.add(User(
        username=body.username,
        full_name=body.full_name,
        password_hash=hash_password(body.password),
        role=body.role,
    ))
    db.commit()
    audit(db, request, admin.username, "create_user", f"user:{body.username}", {"role": body.role})
    return {"username": body.username, "full_name": body.full_name, "role": body.role}


@app.get("/audit")
def audit_log(
    limit: int = Query(default=200, ge=1, le=2000),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    entries = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
    return [
        {
            "timestamp": e.timestamp.isoformat(),
            "username": e.username,
            "action": e.action,
            "resource": e.resource,
            "details": e.details,
            "ip_address": e.ip_address,
        }
        for e in entries
    ]


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------

@app.get("/model-info")
def model_info(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    stages = []
    for stage in STAGES:
        model = registry.get(stage)
        if model:
            tempo = registry.get_tempo(stage)
            stages.append({"available": True, **model.info(), "tempo": tempo.info() if tempo else None})
        else:
            info = {k: v for k, v in STAGE_INFO[stage].items() if not k.endswith("_features")}
            stages.append({"stage": stage, "available": False, **info})
    return {
        "available_stages": registry.available(),
        "tempo_stages": registry.tempo_available(),
        "stages": stages,
        "modality_features": {s: STAGE_INFO[s]["modality_features"] for s in STAGES},
        "scope": SCOPE,
    }


@app.get("/population/risk-distribution")
def population_risk_distribution(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return get_population_risk_distribution()


# ---------------------------------------------------------------------------
# Single-patient tasks
# ---------------------------------------------------------------------------

@app.post("/assessment", status_code=201)
def assessment(
    body: PatientAssessment,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Score a patient along the pathway, explain it with SHAP and save it to the queue."""
    payload = body.model_dump()
    features = model_inputs(payload)
    assessment_id = payload["assessment_id"] or f"ASM-{uuid.uuid4().hex[:12].upper()}"

    if db.query(PatientRecord).filter(PatientRecord.assessment_id == assessment_id).first():
        raise HTTPException(status_code=409, detail=f"Assessment '{assessment_id}' already exists.")

    result = run_pathway(features)
    model = registry.get(result["current_stage"])

    record = PatientRecord(
        assessment_id=assessment_id,
        patient_ref=payload["patient_ref"],
        inputs=features,
        created_by=user.username,
    )
    apply_pathway(record, result)
    db.add(record)
    db.commit()
    db.refresh(record)

    audit(db, request, user.username, "create_assessment", f"assessment:{assessment_id}",
          {"stage": result["current_stage"], "priority": result["priority"]})

    return {
        "id": record.id,
        "assessment_id": assessment_id,
        "patient_ref": payload["patient_ref"],
        **task_for(features),
        "inputs": features,
        "data_quality": calculate_quality(features, model.features, stage_measurements(model.stage)),
        "prediction": {k: result[k] for k in (
            "current_stage", "current_stage_name", "risk_score", "priority", "model", "model_version"
        )},
        "pathway": {k: result[k] for k in ("stages_reached", "trace", "next_step")},
        "decline": result["decline"],
        "explanation": result["explanation"],
        "scope": SCOPE,
    }


@app.post("/predict")
def predict(
    body: PredictionRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Score without saving. Uses the deepest reached stage unless ``stage`` is given."""
    payload = body.model_dump()
    features = model_inputs(payload)
    if payload.get("stage"):
        model = stage_model(payload["stage"])
        result = model.predict(features)
        quality = calculate_quality(features, model.features, stage_measurements(model.stage))
    else:
        pathway = run_pathway(features, explain=False)
        result = {k: pathway[k] for k in ("risk_score", "priority", "model", "model_version")}
        result.update(stage=pathway["current_stage"], trace=pathway["trace"], next_step=pathway["next_step"])
        quality = calculate_quality(
            features, registry.get(pathway["current_stage"]).features,
            stage_measurements(pathway["current_stage"]),
        )
    audit(db, request, user.username, "predict", f"patient:{payload['patient_ref']}")
    return {"patient_ref": payload["patient_ref"], **result, "data_quality": quality, "scope": SCOPE}


@app.post("/explain")
def explain(
    body: PredictionRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    payload = body.model_dump()
    features = model_inputs(payload)
    stage = payload.get("stage") or next(
        (s for s in reversed(reached_stages(features)) if registry.get(s)), None
    )
    model = stage_model(stage)
    audit(db, request, user.username, "explain", f"patient:{payload['patient_ref']}")
    return model.explain(features)


@app.post("/quality")
def quality(body: PredictionRequest, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    payload = body.model_dump()
    features = model_inputs(payload)
    model = registry.get(payload.get("stage") or "stage1")
    return calculate_quality(features, model.features if model else None)


@app.post("/uncertainty")
def uncertainty(body: PredictionRequest, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    payload = body.model_dump()
    return uncertainty_service.estimate(model_inputs(payload), payload.get("stage") or "stage1")


# ---------------------------------------------------------------------------
# Cohort tasks: triage and re-ranking
# ---------------------------------------------------------------------------

@app.post("/rank")
def rank(
    body: RankingRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Rank an ad-hoc list of patients (not saved)."""
    entries = []
    for patient in body.patients:
        payload = patient.model_dump()
        features = model_inputs(payload)
        result = run_pathway(features, explain=False)
        entries.append({
            "assessment_id": payload["assessment_id"],
            "patient_ref": payload["patient_ref"],
            "current_stage": result["current_stage"],
            "risk_score": result["risk_score"],
            "priority": result["priority"],
            "model_version": result["model_version"],
            "next_step": result["next_step"],
            "decline": decline_summary(result["decline"]),
            "data_quality": calculate_quality(
                features, registry.get(result["current_stage"]).features,
                stage_measurements(result["current_stage"]),
            ),
        })
    audit(db, request, user.username, "rank", "adhoc", {"patients": len(entries), "capacity": body.capacity})
    return {"task": "triage", **rank_scored(entries, body.capacity)}


@app.post("/triage")
def triage(
    body: TriageRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Rank the saved patient queue against the available diagnostic capacity."""
    if body.stage is not None and body.stage not in STAGES:
        raise HTTPException(status_code=422, detail=f"Unknown stage '{body.stage}'.")
    result = ranked_queue(db, body.capacity, body.stage)
    audit(db, request, user.username, "triage", "queue",
          {"capacity": body.capacity, "stage": body.stage, "patients": result["total_patients"]})
    return {"task": "triage", "stage_filter": body.stage, **result}


@app.get("/patients")
def patients(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    records = db.query(PatientRecord).order_by(PatientRecord.created_at.desc()).all()
    audit(db, request, user.username, "list_patients", "queue", {"count": len(records)})
    return {
        "total_patients": len(records),
        "patients": [record_to_dict(r) for r in records],
        "scope": SCOPE,
    }


@app.get("/patients/{record_id}")
def patient_detail(
    record_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    record = get_record(db, record_id)
    audit(db, request, user.username, "view_patient", f"assessment:{record.assessment_id}")
    return record_to_dict(record, detail=True)


@app.patch("/patients/{record_id}")
def update_patient(
    record_id: int,
    body: PatientUpdate,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """New-data trigger: add results (e.g. biomarkers, MRI, PET), re-score and re-rank the queue."""
    record = get_record(db, record_id)
    new_values = model_inputs(body.values.model_dump(exclude_none=True))
    if not new_values:
        raise HTTPException(status_code=422, detail="No new values supplied.")

    previous = {"stage": record.current_stage, "priority": record.priority, "risk_score": record.risk_score}
    record.inputs = {**(record.inputs or {}), **new_values}
    apply_pathway(record, run_pathway(record.inputs))
    db.commit()
    db.refresh(record)

    audit(db, request, user.username, "update_patient", f"assessment:{record.assessment_id}",
          {"fields": sorted(new_values), "stage": record.current_stage, "priority": record.priority})

    return {
        "task": "rerank",
        "previous": previous,
        "patient": record_to_dict(record, detail=True),
        "queue": ranked_queue(db, body.capacity),
    }


@app.delete("/patients/{record_id}", status_code=204)
def delete_patient(
    record_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    record = get_record(db, record_id)
    assessment_id = record.assessment_id
    db.delete(record)
    db.commit()
    audit(db, request, user.username, "delete_patient", f"assessment:{assessment_id}")
