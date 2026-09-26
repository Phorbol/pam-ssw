#!/bin/bash
#SBATCH --job-name=s2-formula
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ptsd-primary-2018-20260926/check-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q research/ga_ssw/test_s1_reference.py research/ga_ssw/test_s2_reference.py
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/ptsd-primary-2018-20260926/check_geometry.py
