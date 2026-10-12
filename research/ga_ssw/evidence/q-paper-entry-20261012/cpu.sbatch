#!/bin/bash
#SBATCH --job-name=q-entry-check
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/q-paper-entry-20261012/cpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/q-paper-entry-20261012/cpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/q-paper-entry-20261012/source-cpu-v1
export PYTHONPATH=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/q-paper-entry-20261012/source-cpu-v1
exec /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_descriptor_direction.py tests/standalone/test_recovered_direction_checkpoint.py tests/standalone/test_recovered_direction_driver.py tests/standalone/test_periodic_direction_entry.py tests/standalone/test_constrained_full_direction.py tests/standalone/test_ssw_hookean_entry.py tests/standalone/test_shape_ls_driver.py tests/standalone/test_pool_direction_checkpoint.py
