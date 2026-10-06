#!/usr/bin/env bash
#SBATCH --job-name=m80-dir-xfer
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3%2
#SBATCH --time=00:12:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/m80-direction-transfer-20261007/panel-%A_%a.out

set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
task_dir=research/ga_ssw/evidence/m80-direction-transfer-20261007
exec /home/gengjianrui/.conda/envs/mace_env/bin/python "$task_dir/run.py" \
  --slot "$SLURM_ARRAY_TASK_ID" \
  --output "$task_dir/run-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
