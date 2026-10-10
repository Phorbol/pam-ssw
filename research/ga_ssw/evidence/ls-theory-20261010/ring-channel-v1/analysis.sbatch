#!/bin/bash
#SBATCH --job-name=ls-competing-channels
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:03:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/analysis-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/analysis-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/source:/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/source/research/ga_ssw
exec /home/gengjianrui/.conda/envs/mace_env/bin/python /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/analyze_ls_competing_channels.py /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/channel-v1/run-1728173 /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/run-1728329 /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/analysis-$SLURM_JOB_ID.json
