#!/bin/bash
#SBATCH --job-name=vc-phase139-xfer
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/vc-phase139-transfer-20261012/research/ga_ssw/evidence/vc-phase139-transfer-20261012/slurm-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/vc-phase139-transfer-20261012/research/ga_ssw/evidence/vc-phase139-transfer-20261012/slurm-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

EVIDENCE=/home/gengjianrui/bin/pam-ssw-worktrees/vc-phase139-transfer-20261012/research/ga_ssw/evidence/vc-phase139-transfer-20261012
PYTHON=/home/gengjianrui/.conda/envs/mace_env/bin/python3.12
exec "$PYTHON" "$EVIDENCE/run_phase139.py" --execute --out "$EVIDENCE/run-${SLURM_JOB_ID}"
