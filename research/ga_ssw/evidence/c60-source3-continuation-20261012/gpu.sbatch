#!/bin/bash
#SBATCH --job-name=c60-cont100
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/benchmark-c60-continuation-20261012/research/ga_ssw/evidence/c60-source3-continuation-20261012/gpu-%A_%a.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/benchmark-c60-continuation-20261012/research/ga_ssw/evidence/c60-source3-continuation-20261012/gpu-%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/benchmark-c60-continuation-20261012
TASK_DIR="$ROOT/research/ga_ssw/evidence/c60-source3-continuation-20261012"
OUT_ROOT="${1:?usage: sbatch gpu.sbatch /absolute/path/to/new-output-root}"
cd "$ROOT"
exec /home/gengjianrui/.conda/envs/mace_env/bin/python \
  "$TASK_DIR/runner.py" --output-root "$OUT_ROOT" \
  --arm-index "$SLURM_ARRAY_TASK_ID" --execute
