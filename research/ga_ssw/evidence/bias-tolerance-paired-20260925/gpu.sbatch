#!/bin/bash
# Prepared for review only; this task does not submit it.
#SBATCH --job-name=bias-tol-paired
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=research/ga_ssw/evidence/bias-tolerance-paired-20260925/gpu-%j.out
#SBATCH --error=research/ga_ssw/evidence/bias-tolerance-paired-20260925/gpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONFAULTHANDLER=1
exec /home/gengjianrui/.conda/envs/mace_env/bin/python \
  research/ga_ssw/evidence/bias-tolerance-paired-20260925/run.py --execute
