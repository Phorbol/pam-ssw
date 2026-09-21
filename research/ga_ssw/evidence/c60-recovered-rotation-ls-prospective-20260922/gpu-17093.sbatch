#!/bin/bash
# Prepared only; submission is a main-agent decision after heldout review.
#SBATCH --job-name=c60-rot-ls-17093
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:40:00
#SBATCH --output=gpu-17093-%j.out
set -euo pipefail
HERE="/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-recovered-rotation-ls-prospective-20260922"
cd "$HERE"
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/home/gengjianrui/.conda/envs/mace_env/bin/python
exec "$PYTHON" runner.py --execute --case c60_17093
