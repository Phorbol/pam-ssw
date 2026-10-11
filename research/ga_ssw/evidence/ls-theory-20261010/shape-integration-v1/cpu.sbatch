#!/bin/bash
#SBATCH --job-name=shape-ls-contract
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/shape-integration-v1/cpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/shape-integration-v1/cpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
exec /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_shape_softening.py tests/standalone/test_shape_ls_driver.py tests/standalone/test_ssw_checkpoint.py tests/standalone/test_ls_prequench_settings.py tests/standalone/test_ls_pool_restart.py tests/standalone/test_ssw_hookean_entry.py
