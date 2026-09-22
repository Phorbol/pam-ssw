#!/bin/bash
#SBATCH --job-name=ls-pool-tests
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/ls-pool-restart-20260922/cpu-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_ls_pool_restart.py tests/standalone/test_pool_starter_adapter.py tests/standalone/test_starter_selection.py tests/standalone/test_native_ls_driver.py tests/standalone/test_ls_cycle.py tests/standalone/test_periodic_ls.py tests/standalone/test_ssw_checkpoint.py tests/standalone/test_recovered_direction_checkpoint.py
