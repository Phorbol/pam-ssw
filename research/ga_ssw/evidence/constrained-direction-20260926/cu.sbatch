#!/bin/bash
#SBATCH --job-name=constrained-dir
#SBATCH --partition=CPU-MISC
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/constrained-direction-20260926/cu-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/constrained-direction-20260926/qualify.py cu-fixed --output runs-cu
