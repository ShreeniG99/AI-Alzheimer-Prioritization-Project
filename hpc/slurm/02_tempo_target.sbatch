#!/bin/bash
# Build the TEMPO target (yearly CDR-SB change). Runs alongside the stage jobs.
#SBATCH --job-name=idea-tempo-target
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=hpc/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh

idea_run scripts/10_build_tempo_target.py
