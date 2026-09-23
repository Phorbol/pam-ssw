#!/bin/bash
#SBATCH --job-name=population-c60
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:40:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/population-comparison-20260923/gpu-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-matched-search
export PYTHONNOUSERSITE=1 PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/run_population_comparison.py --plan "$1" --arm "$2" --out "$3"
