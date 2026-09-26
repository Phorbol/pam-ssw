#!/bin/bash
#SBATCH --job-name=vc-readout
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/vc-paper-panel-20260926/research/ga_ssw/evidence/vc-paper-panel-20260926/analysis-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/vc-paper-panel-20260926
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/vc-paper-panel-20260926/analysis.py
