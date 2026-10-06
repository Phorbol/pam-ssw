#!/usr/bin/env bash
#SBATCH --job-name=atomic-exit-regress
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-atomic-budget-release-20261007/regression-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
git rev-parse HEAD
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/test_atomic_climb.py tests/standalone/test_block_ssw.py tests/standalone/test_atomic_climb_rotation_exit.py tests/standalone/test_rotation_exit_policy.py
