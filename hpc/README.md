# Training IDEA on an HPC cluster (SLURM + Apptainer)

The whole training pipeline runs as SLURM batch jobs inside a container, so
the cluster needs nothing installed except Apptainer (Singularity). The
trained models are then copied back to the laptop that runs the app.

## What runs in parallel

```
 01 stages ── array of 4 tasks: one per pathway stage ──┐
   (build dataset + train risk model, 8 cores each)     ├──> 03 TEMPO ── array of 4 tasks
 02 TEMPO target ── 1 task ─────────────────────────────┘
 01 stages ───────────────────────────────────────────────> 04 evaluate + 500-replicate
                                                             bootstrap on 32 cores
```

Three levels of parallelism:

1. **Task parallelism:** the four pathway stages are independent, so they
   train at the same time as a SLURM job array (`--array=1-4`).
2. **Model parallelism:** random forest and XGBoost use every core SLURM
   allocates (`IDEA_N_JOBS` = `$SLURM_CPUS_PER_TASK`).
3. **Data parallelism:** the 500 bootstrap refits are independent and are
   spread over all cores of the node (`--n-jobs`). The resamples are drawn in a
   fixed order first, so the result is identical for any number of cores.

## Measured speed-up

Measured on a 4-core Linux machine with the Python 3.15 stack (wall-clock time):

| Workload | 1 core | 2 cores | 4 cores | Speed-up (4 cores) |
|---|---|---|---|---|
| Bootstrap, Stage 2 random forest, 24 refits | 63.6 s | 30.9 s | 13.9 s | **4.6×** |
| Bootstrap, Stage 1 logistic regression, 200 refits | 4.8 s | 3.9 s | 2.9 s | 1.7× |

- Heavy models (random forest) scale almost perfectly. 500 random-forest refits
  take about 22 minutes on one core and should take under a minute on a
  32-core node.
- Logistic regression is so fast per refit that scheduling overhead dominates,
  so it gains less. This is expected and worth stating honestly.
- Parallel results are identical to the sequential run: the largest difference
  over all 500 × 1,677 bootstrap predictions was exactly 0.0.

## Step by step

### 1. Copy the code and data to the cluster

```bash
git clone https://github.com/ShreeniG99/AI-Alzheimer-Prioritization-Project.git idea
cd idea
mkdir -p ~/idea/data/raw
# copy the ADNI CSV files into ~/idea/data/raw (scp / rsync from your laptop)
```

ADNI data may only be stored where your ADNI Data Use Agreement allows. Check
that the cluster storage qualifies, and delete the data when you are done.

### 2. Build the container (once)

```bash
apptainer build idea-train.sif hpc/idea-train.def       # add --fakeroot if required
mv idea-train.sif ~/idea/
```

If you cannot build on the cluster, build the Docker image on a machine with
Docker (`docker build -f Dockerfile.train -t idea-train .`), then convert it:
`apptainer build idea-train.sif docker-daemon://idea-train:latest`.

### 3. Check the paths

Open `hpc/env.sh` and confirm `IDEA_SIF`, `IDEA_RAW` and `IDEA_WORK`. You may
also need to add your cluster's partition or account to the `#SBATCH` lines in
`hpc/slurm/*.sbatch`, for example `#SBATCH --partition=cpu`.

### 4. Submit everything

```bash
bash hpc/submit_all.sh
squeue -u $USER          # watch the jobs
tail -f hpc/logs/*.out   # follow the logs
```

### 5. Bring the trained models back to the laptop

```bash
cd ~/idea/work
zip -r idea-trained-models.zip artifacts reports data/processed
```

Copy `idea-trained-models.zip` to the laptop and unzip it in the project
folder. The app (local or Docker) picks the models up on its next start.

## Files

| File | Purpose |
|---|---|
| `Dockerfile.train` | Training image (Docker) |
| `hpc/idea-train.def` | Same image for Apptainer |
| `hpc/env.sh` | Paths and the `idea_run` helper used by every job |
| `hpc/slurm/01_stages.sbatch` | Array job: dataset + risk model per stage |
| `hpc/slurm/02_tempo_target.sbatch` | TEMPO target |
| `hpc/slurm/03_tempo.sbatch` | Array job: TEMPO model per stage |
| `hpc/slurm/04_evaluate.sbatch` | Evaluation + parallel bootstrap |
| `hpc/submit_all.sh` | Submits all jobs with dependencies |
