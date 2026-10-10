#!/bin/bash
#SBATCH --job-name=ls-escape-analysis
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:03:00
#SBATCH --dependency=afterany:1728766
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/single-escape-v1/analysis-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/single-escape-v1/analysis-%j.err
set -euo pipefail
exec /home/gengjianrui/.conda/envs/mace_env/bin/python /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/single-escape-v1/analyze.py /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/single-escape-v1/run-1728766/summary.json /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-theory-20261010/single-escape-v1/analysis.json
