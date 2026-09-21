#!/bin/bash
#SBATCH --job-name=recrot-ls-contract
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/recovered-rotation-ls-contract-20260922/slurm-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q \
  tests/standalone/test_recovered_rotation_ls.py \
  > research/ga_ssw/evidence/recovered-rotation-ls-contract-20260922/pytest-${SLURM_JOB_ID}.txt 2>&1
