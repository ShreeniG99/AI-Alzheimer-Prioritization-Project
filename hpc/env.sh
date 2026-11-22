# Shared settings for the IDEA SLURM jobs. Edit the three paths, then submit
# with hpc/submit_all.sh from the repository root.

# Container image built from hpc/idea-train.def
IDEA_SIF="${IDEA_SIF:-$HOME/idea/idea-train.sif}"
# Folder with the ADNI CSV files (read-only). Keep it on storage your ADNI DUA allows.
IDEA_RAW="${IDEA_RAW:-$HOME/idea/data/raw}"
# Output folder: data/processed, artifacts and reports are written here.
IDEA_WORK="${IDEA_WORK:-$HOME/idea/work}"

mkdir -p "$IDEA_WORK/data/processed" "$IDEA_WORK/artifacts" "$IDEA_WORK/reports"

# Run a pipeline script inside the container. Every tree model and the
# bootstrap use as many workers as SLURM gave this task.
idea_run() {
    export APPTAINERENV_IDEA_N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
    export APPTAINERENV_ALZ_RAW_DIR=/data/raw
    export APPTAINERENV_ALZ_PROCESSED_DIR=/work/data/processed
    export APPTAINERENV_ALZ_ARTIFACTS_DIR=/work/artifacts
    export APPTAINERENV_ALZ_REPORTS_DIR=/work/reports
    apptainer exec --cleanenv \
        --bind "$IDEA_RAW:/data/raw:ro" \
        --bind "$IDEA_WORK:/work" \
        --pwd /app \
        "$IDEA_SIF" python "$@"
}
