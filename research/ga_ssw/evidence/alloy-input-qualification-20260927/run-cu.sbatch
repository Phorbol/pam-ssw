#!/bin/bash
#SBATCH --job-name=cuau-input-qualification
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=research/ga_ssw/evidence/alloy-input-qualification-20260927/gpu-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
p=research/ga_ssw/evidence/alloy-input-qualification-20260927
/home/gengjianrui/.conda/envs/mace_env/bin/python "$p/qualify.py" --initial "$p/Cu30Au30_gupta_source.extxyz" --system Cu30Au30 --out "$p/run-$SLURM_JOB_ID"
