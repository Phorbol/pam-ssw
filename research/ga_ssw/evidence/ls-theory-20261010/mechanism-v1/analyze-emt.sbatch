#!/bin/bash
#SBATCH --job-name=ls-mechanism-analysis
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:03:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/analysis-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/analysis-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
exec /home/gengjianrui/.conda/envs/mace_env/bin/python /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/analyze_ls_response.py /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/emt-analysis-$SLURM_JOB_ID.json /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/emt-1728035/case-0 /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/mechanism-v1/emt-1728035/case-1
