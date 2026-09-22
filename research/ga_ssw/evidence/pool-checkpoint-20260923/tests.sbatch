#!/bin/bash
#SBATCH --job-name=pool-checkpoint
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/pool-checkpoint-20260923/cpu-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_pool_checkpoint_real.py tests/standalone/test_pool_checkpoint.py tests/standalone/test_pool_direction_checkpoint.py tests/standalone/test_pool_starter_adapter.py tests/standalone/test_ssw_checkpoint.py tests/standalone/test_recovered_direction_checkpoint.py tests/standalone/test_starter_selection.py tests/standalone/test_ls_pool_restart.py
