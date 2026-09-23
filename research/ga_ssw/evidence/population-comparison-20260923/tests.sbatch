#!/bin/bash
#SBATCH --job-name=population-tests
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/population-comparison-20260923/tests-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-matched-search
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_population_comparison_runner.py tests/standalone/test_ga_checkpoint.py tests/standalone/test_ga_walker_options.py
