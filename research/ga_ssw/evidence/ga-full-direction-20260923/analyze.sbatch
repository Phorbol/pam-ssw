#!/bin/bash
#SBATCH --job-name=ga-direction-audit
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:02:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/ga-full-direction-20260923/audit-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/ga-full-direction-20260923/analyze.py
