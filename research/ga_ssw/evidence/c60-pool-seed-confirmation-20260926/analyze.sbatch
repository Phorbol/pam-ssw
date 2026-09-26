#!/bin/bash
#SBATCH --job-name=ls-pool-readout
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-pool-seed-confirmation-20260926/readout-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
analysis_root=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
panel_root=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-pool-seed-confirmation-20260926
export PYTHONPATH="$analysis_root"
cd "$analysis_root"
git rev-parse HEAD
sha256sum research/ga_ssw/analyze_ls_pool_routing_panel.py
/home/gengjianrui/.conda/envs/mace_env/bin/python -u research/ga_ssw/analyze_ls_pool_routing_panel.py \
 --plan "$panel_root/plan.json" --evidence-root "$panel_root/view-1501032" \
 --output-dir "$panel_root/readout-1501032"
