"""Shared fixtures.

The whole pipeline runs once per test session on SYNTHETIC data in a
temporary directory (never the real data/ or artifacts/ folders), then the
API is exercised against those artifacts and a temporary SQLite database.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["stage1", "stage2", "stage3", "stage4"]


def run(script: str, *args: str, env: dict) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("pipeline")
    dirs = {
        "ALZ_RAW_DIR": base / "raw",
        "ALZ_PROCESSED_DIR": base / "processed",
        "ALZ_ARTIFACTS_DIR": base / "artifacts",
        "ALZ_REPORTS_DIR": base / "reports",
    }
    env = {**os.environ, **{k: str(v) for k, v in dirs.items()}, "MPLBACKEND": "Agg"}

    run("00_make_synthetic_data.py", "--out", str(dirs["ALZ_RAW_DIR"]), "--subjects", "1200", env=env)
    for stage in STAGES:
        run("02_build_dataset.py", "--stage", stage, env=env)
        run("03_train.py", "--stage", stage, env=env)
    run("08_uncertainty_analysis.py", "--stage", "stage1", "--replicates", "20", env=env)
    run("05_evaluate_prioritization.py", env=env)

    os.environ.update({k: str(v) for k, v in dirs.items()})
    os.environ["DATABASE_URL"] = f"sqlite:///{base / 'test.db'}"
    os.environ["JWT_SECRET"] = "test-secret"
    os.environ["ADMIN_USERNAME"] = "admin"
    os.environ["ADMIN_PASSWORD"] = "admin-password"
    return dirs


@pytest.fixture(scope="session")
def client(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def admin_headers(client):
    response = client.post("/auth/login", json={"username": "admin", "password": "admin-password"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
