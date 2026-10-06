#!/bin/bash
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-atomic-budget-release-20261007/direct-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-atomic-budget-release-20261007/direct-%j.err

set -euo pipefail
CHECKOUT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
cd "${CHECKOUT}"
SCRIPT_DIR="${CHECKOUT}/research/ga_ssw/evidence/tio2-atomic-budget-release-20261007"
PYTHON=/home/gengjianrui/.conda/envs/mace_env/bin/python3.12
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
OUTPUT_ROOT="${TIO2_ATOMIC_RELEASE_OUT:-${SCRIPT_DIR}}"
"${PYTHON}" "${SCRIPT_DIR}/direct_quench.py" --execute-all \
  --out-root "${OUTPUT_ROOT}" --run-id "${SLURM_JOB_ID}"
