#!/bin/bash
#SBATCH --job-name=population-audit
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/population-comparison-20260923/audit-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/analyze_population_comparison.py --root research/ga_ssw/evidence/population-comparison-20260923/runs --output research/ga_ssw/evidence/population-comparison-20260923/analysis-interim-seed3.json --expected-runs 6
