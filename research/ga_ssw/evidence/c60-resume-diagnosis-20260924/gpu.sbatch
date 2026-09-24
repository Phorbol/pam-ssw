#!/bin/bash
#SBATCH --job-name=c60-resume-diagnosis
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-resume-diagnosis-20260924/gpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-resume-diagnosis-20260924/gpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONPATH="$PWD/research/ga_ssw/evidence/c60-long-budget-20260924/source"
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_c60_long_budget.py
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/c60-resume-diagnosis-20260924/run.py
