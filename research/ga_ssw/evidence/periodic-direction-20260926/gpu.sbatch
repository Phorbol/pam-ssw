#!/bin/bash
#SBATCH --job-name=periodic-dir
#SBATCH --partition=8V100V0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:45:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/periodic-direction-20260926/gpu-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/periodic-direction-20260926/qualify.py
