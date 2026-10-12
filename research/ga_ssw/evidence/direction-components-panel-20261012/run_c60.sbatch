#!/bin/bash
# Submit only after the zero-PES preflight and review.
# Corrected run: one V100, 29 minutes, retaining the 30-minute series ceiling.
#SBATCH --job-name=dir-components-c60
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:29:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/direction-components-panel-20261012/c60-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/direction-components-panel-20261012/c60-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
TASK_DIR="$ROOT/research/ga_ssw/evidence/direction-components-panel-20261012"
OUT="$TASK_DIR/runs-${SLURM_JOB_ID}"
START_SECONDS=$SECONDS
SEARCH_WINDOW_SECONDS=1620
cd "$ROOT"
for METHOD in full local_only feedback_only; do
  REMAINING=$((SEARCH_WINDOW_SECONDS - (SECONDS - START_SECONDS)))
  if (( REMAINING <= 0 )); then
    echo "C60 search window exhausted; remaining methods stay unrun" >&2
    break
  fi
  /home/gengjianrui/.conda/envs/mace_env/bin/python "$TASK_DIR/run.py" --run \
    --case c60 --method "$METHOD" --out "$OUT" --wall-seconds "$REMAINING"
done
