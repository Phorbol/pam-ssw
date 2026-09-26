#!/bin/bash
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --job-name=c60-path-readout
#SBATCH --output=research/ga_ssw/evidence/c60-path-diagnostic-20260927/readout-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
p=research/ga_ssw/evidence/c60-path-diagnostic-20260927
out="$p/verified-$SLURM_JOB_ID"
/home/gengjianrui/.conda/envs/mace_env/bin/python "$p/analyze_path.py" --run-dir "$p/run-1504112" --out-dir "$out"
/home/gengjianrui/.conda/envs/mace_env/bin/python "$p/analyze_frozen_ls.py" --run-dir "$p/run-1504112" --out "$out/frozen-ls.json"
