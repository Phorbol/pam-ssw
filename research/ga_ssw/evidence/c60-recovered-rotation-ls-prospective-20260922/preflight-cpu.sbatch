#!/bin/bash
# One zero-PES integrity preflight; never submit with --execute.
#SBATCH --job-name=c60-rot-ls-preflight
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=preflight-%j.out
set -euo pipefail
HERE="/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-recovered-rotation-ls-prospective-20260922"
cd "$HERE"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$HERE/.pycache"
PYTHON=/home/gengjianrui/.conda/envs/mace_env/bin/python
"$PYTHON" -B -m py_compile runner.py preflight.py ledger.py validator.py
"$PYTHON" preflight.py
