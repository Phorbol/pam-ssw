#!/bin/bash
#SBATCH --job-name=c60-dir-qual
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-direction-transfer-20261007/prepare-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-direction-transfer-20261007/prepare-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
TASK_DIR="$ROOT/research/ga_ssw/evidence/c60-direction-transfer-20261007"
cd "$ROOT"
exec /home/gengjianrui/.conda/envs/mace_env/bin/python "$TASK_DIR/prepare.py" \
  --execute --output "$TASK_DIR/prepared-${SLURM_JOB_ID}"
