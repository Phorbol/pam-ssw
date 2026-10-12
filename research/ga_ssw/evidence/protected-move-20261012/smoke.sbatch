#!/bin/bash
#SBATCH --job-name=protected-move-smoke
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/protected-move-20261012/smoke-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/protected-move-20261012/smoke-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$1"
export PYTHONPATH="$1"
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/probe_protected_move_paper_cases.py /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/protected-move-20261012/smoke-$SLURM_JOB_ID --data-root /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification --smoke
