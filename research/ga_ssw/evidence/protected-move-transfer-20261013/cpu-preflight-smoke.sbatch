#!/bin/bash
#SBATCH --job-name=pm-transfer-check
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/protected-move-transfer-20261013/cpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/protected-move-transfer-20261013/cpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$1"
export PYTHONPATH="$1"
PY=/home/gengjianrui/.conda/envs/mace_env/bin/python
DATA_ROOT="${2:-/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification}"
EVIDENCE="$DATA_ROOT/research/ga_ssw/evidence/protected-move-transfer-20261013"
"$PY" research/ga_ssw/probe_protected_move_transfer.py "$EVIDENCE/preflight-$SLURM_JOB_ID" --data-root "$DATA_ROOT"
"$PY" research/ga_ssw/probe_protected_move_transfer.py "$EVIDENCE/smoke-$SLURM_JOB_ID" --data-root "$DATA_ROOT" --smoke
