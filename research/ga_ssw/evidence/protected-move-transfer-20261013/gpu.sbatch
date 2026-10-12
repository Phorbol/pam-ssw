#!/bin/bash
#SBATCH --job-name=pm-transfer-gpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/protected-move-transfer-20261013/gpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/protected-move-transfer-20261013/gpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$1"
export PYTHONPATH="$1"
DATA_ROOT="${2:-/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification}"
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/probe_protected_move_transfer.py "$DATA_ROOT/research/ga_ssw/evidence/protected-move-transfer-20261013/run-$SLURM_JOB_ID" --data-root "$DATA_ROOT" --execute
