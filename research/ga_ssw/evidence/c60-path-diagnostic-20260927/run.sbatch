#!/bin/bash
#SBATCH --job-name=c60-path-diagnostic
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=research/ga_ssw/evidence/c60-path-diagnostic-20260927/gpu-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
probe_dir=research/ga_ssw/evidence/c60-path-diagnostic-20260927
/home/gengjianrui/.conda/envs/mace_env/bin/python "$probe_dir/run_neb.py" --initial "$probe_dir/inputs/initial.extxyz" --final "$probe_dir/inputs/final.extxyz" --out "$probe_dir/run-$SLURM_JOB_ID" --call-cap 3000 --deadline-seconds 1080
