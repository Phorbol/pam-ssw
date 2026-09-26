#!/bin/bash
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:03:00
#SBATCH --job-name=c60-direction-readout
#SBATCH --output=research/ga_ssw/evidence/c60-path-diagnostic-20260927/directions-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
p=research/ga_ssw/evidence/c60-path-diagnostic-20260927
/home/gengjianrui/.conda/envs/mace_env/bin/python "$p/analyze_saved_directions.py" --out "$p/directions-$SLURM_JOB_ID.json"
