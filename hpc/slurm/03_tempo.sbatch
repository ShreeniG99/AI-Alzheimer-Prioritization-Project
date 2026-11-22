#!/bin/bash
# Train the TEMPO decline-rate model for all four stages in parallel.
#SBATCH --job-name=idea-tempo
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh

STAGE="stage${SLURM_ARRAY_TASK_ID}"
idea_run scripts/11_train_tempo.py --stage "$STAGE"
