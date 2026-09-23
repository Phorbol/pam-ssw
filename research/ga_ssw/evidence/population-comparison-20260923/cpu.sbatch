#!/bin/bash
#SBATCH --job-name=population-cu13
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/population-comparison-20260923/cpu-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-matched-search
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/run_population_comparison.py --plan "$1" --arm "$2" --out "$3"
