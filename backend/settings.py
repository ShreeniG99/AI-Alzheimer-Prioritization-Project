"""Runtime configuration, read from environment variables."""

import logging
import os
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("alzheimer.settings")


def _path(env_var: str, default: str) -> Path:
    value = os.getenv(env_var)
    return Path(value) if value else PROJECT_ROOT / default


DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'app.db'}")

ARTIFACTS_DIR = _path("ALZ_ARTIFACTS_DIR", "artifacts")
REPORTS_DIR = _path("ALZ_REPORTS_DIR", "reports")
PROCESSED_DIR = _path("ALZ_PROCESSED_DIR", "data/processed")

JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    # Development fallback: tokens stop working when the process restarts.
    JWT_SECRET = secrets.token_urlsafe(48)
    logger.warning("JWT_SECRET is not set; using a random per-process secret (development only).")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

# Optional first administrator, created on startup only if no users exist yet.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

SCOPE = (
    "Research clinical decision-support prototype; "
    "no diagnosis or treatment recommendation."
)
