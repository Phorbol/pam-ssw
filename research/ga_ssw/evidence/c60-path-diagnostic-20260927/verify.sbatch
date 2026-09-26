#!/bin/bash
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:03:00
#SBATCH --job-name=c60-saved-verify
#SBATCH --output=research/ga_ssw/evidence/c60-path-diagnostic-20260927/verify-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/c60-path-diagnostic-20260927/verify_saved.py
