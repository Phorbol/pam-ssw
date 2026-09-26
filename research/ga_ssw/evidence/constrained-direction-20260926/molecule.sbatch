#!/bin/bash
#SBATCH --job-name=constrained-dir
#SBATCH --partition=4V100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:20:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/constrained-direction-20260926/molecule-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/constrained-direction-20260926/qualify.py c4h6-point --output runs-molecule
