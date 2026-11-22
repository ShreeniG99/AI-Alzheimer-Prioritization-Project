#!/bin/bash
# Evaluate prioritisation and run the 500-replicate bootstrap, with the
# refits spread over every allocated core.
#SBATCH --job-name=idea-evaluate
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh

# one thread per refit worker, so the cores are not oversubscribed
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1

idea_run scripts/05_evaluate_prioritization.py
idea_run scripts/08_uncertainty_analysis.py --stage stage1 --replicates 500 --n-jobs "$SLURM_CPUS_PER_TASK"
