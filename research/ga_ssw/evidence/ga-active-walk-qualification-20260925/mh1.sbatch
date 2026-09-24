#!/bin/bash
#SBATCH --job-name=ga-mh1-recovery
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=research/ga_ssw/evidence/ga-active-walk-qualification-20260925/mh1-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
probe=research/ga_ssw/evidence/ga-active-walk-qualification-20260925/mh1_probe.py
python_bin=/home/gengjianrui/.conda/envs/mace_env/bin/python
export probe python_bin
timeout 29m bash -euc '
"$python_bin" "$probe" full
"$python_bin" "$probe" pause
"$python_bin" "$probe" resume
"$python_bin" "$probe" analyze
'
