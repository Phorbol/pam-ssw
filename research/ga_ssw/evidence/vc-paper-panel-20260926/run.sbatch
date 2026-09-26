#!/bin/bash
# Prepared only. Root submits after reviewing the frozen runner and budget.
#SBATCH --job-name=vc-cell-pair
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/vc-paper-panel-20260926/research/ga_ssw/evidence/vc-paper-panel-20260926/slurm-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/vc-paper-panel-20260926/research/ga_ssw/evidence/vc-paper-panel-20260926/slurm-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/gengjianrui/bin/pam-ssw-worktrees/vc-paper-panel-20260926
export PYTHONPATH="$PWD"
model=${MACE_MODEL:?set MACE_MODEL to the frozen MACE-OMAT-0-small model file}
input=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/literature/benchmark-sources/coordinates/rutile.extxyz
out=research/ga_ssw/evidence/vc-paper-panel-20260926/run-${SLURM_JOB_ID:?}
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/run_vc_mechanism_pair.py \
  --input "$input" \
  --backend mace \
  --model "$model" \
  --device cuda \
  --out "$out" \
  --seed 17 \
  --steps 2 \
  --cell-cycles 5 \
  --request-cap-per-arm 6000 \
  --seconds-per-arm 600 \
  --gate-request-cap 300 \
  --gate-seconds 300 \
  --atomic-gaussians 10 \
  --rotation-hvp 100 \
  --cell-rotation-requests 6
