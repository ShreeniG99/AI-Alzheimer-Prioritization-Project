"""
Bootstrap uncertainty analysis.

Refits the selected model architecture (same preprocessing + estimator,
without calibration) on bootstrap resamples of the stage dataset and stores
each refit's raw probability for every cohort row:

    artifacts/<stage>_bootstrap_predictions.npz
        predictions : (replicates, rows) float32 raw model probabilities
        row_indices : (rows,) row positions in data/processed/<stage>.csv

    reports/<stage>_bootstrap_summary.csv
        per-row mean / median / 2.5% / 97.5% / std

The backend uncertainty endpoint reads the .npz file. Intervals describe
model-estimate variability in the retrospective research cohort only; they are
not clinical confidence intervals.

Usage:
    python scripts/08_uncertainty_analysis.py --stage stage1 --replicates 500
"""

import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from common import artifacts_dir, load_config, processed_dir, reports_dir


def _fit_replicate(pipeline, X, y, idx):
    model = clone(pipeline)
    model.fit(X.iloc[idx], y[idx])
    return model.predict_proba(X)[:, 1].astype(np.float32)


def main(stage: str, replicates: int, seed: int, n_jobs: int = 1) -> None:
    model_path = artifacts_dir() / f"{stage}_model.joblib"
    data_path = processed_dir() / f"{stage}.csv"

    if not model_path.exists():
        raise SystemExit(f"{model_path} does not exist. Run 03_train.py first.")
    if not data_path.exists():
        raise SystemExit(f"{data_path} does not exist. Run 02_build_dataset.py first.")

    bundle = joblib.load(model_path)
    features = bundle["features"]
    data = pd.read_csv(data_path)
    X = data[features]
    y = data["target"].astype(int).to_numpy()

    # Draw every resample first, in the same order as a sequential run, so the
    # result is identical whatever the number of parallel workers.
    rng = np.random.default_rng(seed)
    samples = []
    attempts = 0
    while len(samples) < replicates:
        attempts += 1
        if attempts > replicates * 5:
            raise RuntimeError("Too many bootstrap resamples without both classes.")
        idx = rng.integers(0, len(data), size=len(data))
        if len(np.unique(y[idx])) < 2:
            continue
        samples.append(idx)

    # Each refit is independent, so they run in parallel (one per CPU core on HPC).
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_replicate)(bundle["pipeline"], X, y, idx) for idx in samples
    )
    elapsed = time.perf_counter() - start
    predictions = np.vstack(results).astype(np.float32)
    print(f"Bootstrap refits: {replicates} in {elapsed:.1f} s using n_jobs={n_jobs}")

    artifacts_dir().mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir() / f"{stage}_bootstrap_predictions.npz"
    np.savez_compressed(
        out_path,
        predictions=predictions,
        row_indices=np.arange(len(data)),
        model=np.asarray(bundle["selected_model"]),
    )

    summary = pd.DataFrame({
        "row_index": np.arange(len(data)),
        "RID": data["RID"] if "RID" in data.columns else np.nan,
        "target": y,
        "bootstrap_mean": predictions.mean(axis=0),
        "bootstrap_median": np.median(predictions, axis=0),
        "lower_95": np.percentile(predictions, 2.5, axis=0),
        "upper_95": np.percentile(predictions, 97.5, axis=0),
        "std": predictions.std(axis=0, ddof=1),
    })
    reports_dir().mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports_dir() / f"{stage}_bootstrap_summary.csv", index=False)

    width = summary["upper_95"] - summary["lower_95"]
    print(f"\nSaved {out_path}")
    print(f"Model: {bundle['selected_model']}  replicates: {replicates}  rows: {len(data)}")
    print(f"Median 95% interval width: {width.median():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(load_config()["features"]), default="stage1")
    ap.add_argument("--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-jobs", type=int,
        default=int(os.environ.get("IDEA_N_JOBS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1),
        help="Parallel workers for the refits (default: $IDEA_N_JOBS, $SLURM_CPUS_PER_TASK or 1).",
    )
    args = ap.parse_args()
    main(args.stage, args.replicates, args.seed, args.n_jobs)
