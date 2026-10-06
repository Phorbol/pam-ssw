#!/bin/bash
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-atomic-budget-release-20261007/direct-readout-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
task_dir=research/ga_ssw/evidence/tio2-atomic-budget-release-20261007
git rev-parse HEAD
/home/gengjianrui/.conda/envs/mace_env/bin/python "$task_dir/compare_direct.py" \
  "$task_dir/direct-${1:?GPU ID}-0" "$task_dir/direct-${1}-1" --out "$task_dir/direct-readout-${SLURM_JOB_ID}"
