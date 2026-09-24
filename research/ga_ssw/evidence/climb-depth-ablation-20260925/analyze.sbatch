#!/bin/bash
#SBATCH --job-name=climb-depth-readout
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=research/ga_ssw/evidence/climb-depth-ablation-20260925/analysis-%j.out
#SBATCH --error=research/ga_ssw/evidence/climb-depth-ablation-20260925/analysis-%j.err
set -eu
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/climb-depth-ablation-20260925/analyze.py
