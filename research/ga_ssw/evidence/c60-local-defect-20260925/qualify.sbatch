#!/bin/bash
#SBATCH --job-name=c60-defect-qualify
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=research/ga_ssw/evidence/c60-local-defect-20260925/qualify-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
timeout 9m /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/c60-local-defect-20260925/qualify.py
