#!/usr/bin/env bash
#SBATCH --job-name=m80-dir-readout
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/m80-direction-transfer-20261007/readout-%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
task_dir=research/ga_ssw/evidence/m80-direction-transfer-20261007
/home/gengjianrui/.conda/envs/mace_env/bin/python "$task_dir/analyze.py" \
  "${1:?pass slot 0 output directory}" "${2:?pass slot 1 output directory}" \
  "${3:?pass slot 2 output directory}" "${4:?pass slot 3 output directory}" \
  --output "$task_dir/readout-${SLURM_JOB_ID}"
