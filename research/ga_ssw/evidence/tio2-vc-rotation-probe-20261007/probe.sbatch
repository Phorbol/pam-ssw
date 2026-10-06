#!/usr/bin/env bash
#SBATCH --job-name=tio2-vc-rotation
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-vc-rotation-probe-20261007/probe-%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
task_dir=research/ga_ssw/evidence/tio2-vc-rotation-probe-20261007
exec /home/gengjianrui/.conda/envs/mace_env/bin/python3.12 "$task_dir/run.py" \
  --execute --output "$task_dir/run-${SLURM_JOB_ID}"
