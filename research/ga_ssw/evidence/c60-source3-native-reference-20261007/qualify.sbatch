#!/bin/bash
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-native-reference-20261007/qualify-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_DIR=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-native-reference-20261007
PREPARED="${1:?prepared directory}"
READOUT="${2:?readout directory}"
/home/gengjianrui/.conda/envs/mace_env/bin/python "$PREPARED/qualify.py" --prepared "$PREPARED" --readout "$READOUT" --output "$TASK_DIR/qualified-${SLURM_JOB_ID}"
