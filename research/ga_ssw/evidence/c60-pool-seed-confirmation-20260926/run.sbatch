#!/bin/bash
#SBATCH --job-name=c60-pool-confirm
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-pool-seed-confirmation-20260926/gpu-%A_%a.out
set -euo pipefail
main_root=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
worker_root=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
panel_root="$main_root/research/ga_ssw/evidence/c60-pool-seed-confirmation-20260926"
methods=(mc uniform pam)
case_index=0
method_index=$((SLURM_ARRAY_TASK_ID % 3))
export PYTHONNOUSERSITE=1 PYTHONPATH="$main_root" PAMSSW_MAINLINE="$main_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$main_root"
# No automatic retry. Raw partial progress/ledger remains after timeout.
timeout --signal=TERM --kill-after=5s 1790s \
  /home/gengjianrui/.conda/envs/mace_env/bin/python -u \
  "$worker_root/research/ga_ssw/run_ls_pool_routing_panel.py" \
  --plan "$panel_root/plan.json" --case-index "$case_index" \
  --mode "${methods[$method_index]}" \
  --out "$panel_root/run-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
