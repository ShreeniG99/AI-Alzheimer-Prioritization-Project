#!/bin/bash
# Submit the whole IDEA training pipeline to SLURM with job dependencies.
# Run from the repository root:  bash hpc/submit_all.sh
#
#   01 stages (4 tasks) ──┐
#                         ├──> 03 TEMPO (4 tasks)
#   02 TEMPO target ──────┘
#   01 stages ───────────────> 04 evaluate + parallel bootstrap
set -euo pipefail
mkdir -p hpc/logs

stages=$(sbatch --parsable hpc/slurm/01_stages.sbatch)
target=$(sbatch --parsable hpc/slurm/02_tempo_target.sbatch)
tempo=$(sbatch --parsable --dependency=afterok:${stages}:${target} hpc/slurm/03_tempo.sbatch)
evaluate=$(sbatch --parsable --dependency=afterok:${stages} hpc/slurm/04_evaluate.sbatch)

echo "Submitted: stages=$stages  tempo-target=$target  tempo=$tempo  evaluate=$evaluate"
echo "Watch with:  squeue -u \$USER     Logs: hpc/logs/"
