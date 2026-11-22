#!/bin/bash
# Build the dataset and train the risk model for all four pathway stages
# in parallel: one array task per stage.
#SBATCH --job-name=idea-stages
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh

STAGE="stage${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] $STAGE on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs"
idea_run scripts/02_build_dataset.py --stage "$STAGE"
idea_run scripts/03_train.py --stage "$STAGE"
echo "[$(date)] $STAGE done"
