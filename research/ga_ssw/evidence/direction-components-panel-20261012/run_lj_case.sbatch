#!/bin/bash
# Submit only after the zero-PES preflight and review.
# Usage: sbatch run_lj_case.sbatch lj38|lj55
#SBATCH --job-name=dir-components-lj
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/direction-components-panel-20261012/lj-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
TASK_DIR="$ROOT/research/ga_ssw/evidence/direction-components-panel-20261012"
CASE="${1:?usage: sbatch run_lj_case.sbatch lj38|lj55}"
case "$CASE" in lj38|lj55) ;; *) echo "invalid LJ case: $CASE" >&2; exit 2 ;; esac
OUT="$TASK_DIR/runs-${SLURM_JOB_ID}"
START_SECONDS=$SECONDS
SEARCH_WINDOW_SECONDS=540
cd "$ROOT"
for METHOD in full local_only feedback_only; do
  REMAINING=$((SEARCH_WINDOW_SECONDS - (SECONDS - START_SECONDS)))
  if (( REMAINING <= 0 )); then
    echo "LJ search window exhausted; remaining methods stay unrun" >&2
    break
  fi
  /home/gengjianrui/.conda/envs/mace_env/bin/python "$TASK_DIR/run.py" --run \
    --case "$CASE" --method "$METHOD" --out "$OUT" --wall-seconds "$REMAINING"
done
