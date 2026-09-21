#!/bin/bash
# Prepared only; root submits after zero-PES preflight and review.
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=01:40:00
#SBATCH --job-name=c60-hold-r-17096
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/vc-qualification-audit/research/ga_ssw/evidence/c60-rotation-heldout-20260921/gpu-recovered-17096-%j.log
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/gengjianrui/bin/pam-ssw-worktrees/vc-qualification-audit/research/ga_ssw/evidence/c60-rotation-heldout-20260921
/home/gengjianrui/.conda/envs/mace_env/bin/python runner.py --execute --arm recovered_rotation --case c60_17096
