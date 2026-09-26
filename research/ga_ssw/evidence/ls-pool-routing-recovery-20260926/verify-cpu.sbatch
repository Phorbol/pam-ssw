#!/bin/bash
#SBATCH --job-name=ls-pool-recovery
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/ls-pool-routing-recovery-20260926/cpu-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
cd "$PYTHONPATH"
panel=research/ga_ssw/evidence/ls-pool-routing-recovery-20260926
/home/gengjianrui/.conda/envs/mace_env/bin/python "$panel/analyzer-59c85f7.py" \
 --evidence-root "$panel/derived-view" \
 --plan /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-pool-routing-20260926/plan.json \
 --output-dir "$panel/readout-cpu-${SLURM_JOB_ID}"
