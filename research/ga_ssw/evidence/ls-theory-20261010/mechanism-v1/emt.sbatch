#!/bin/bash
#SBATCH --job-name=ls-mechanism-emt
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/emt-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/emt-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/source
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/source
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/probe_ls_response.py emt /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/emt-$SLURM_JOB_ID
