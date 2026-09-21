#!/bin/bash
#SBATCH --job-name=c60-heldout-readout-test
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/vc-qualification-audit/research/ga_ssw/evidence/c60-heldout-readout-20260922/test-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/vc-qualification-audit/research/ga_ssw/evidence/c60-heldout-readout-20260922/test-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/vc-qualification-audit/research/ga_ssw/evidence/c60-heldout-readout-20260922
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q test_readout.py
