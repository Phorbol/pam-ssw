#!/bin/bash
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-native-reference-20261007/readout-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_DIR=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-native-reference-20261007
PREPARED="${1:?prepared directory}"
/home/gengjianrui/.conda/envs/mace_env/bin/python "$PREPARED/analyze.py" --prepared "$PREPARED" --output "$TASK_DIR/readout-${SLURM_JOB_ID}"
