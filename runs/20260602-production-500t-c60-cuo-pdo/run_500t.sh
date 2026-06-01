#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/tmp/SSW-worktrees/direct-qp-ssw/runs/20260602-production-500t-c60-cuo-pdo"
cd /tmp/SSW-worktrees/direct-qp-ssw

exec > "${RUN_DIR}/production500.log" 2>&1
echo "$$" > "${RUN_DIR}/pid.txt"
echo "started $(date -Is)"

eval "$(mamba shell hook --shell bash)"
mamba activate mace_les

python "${RUN_DIR}/run_matrix.py" \
  --systems c60,cuo,pdo \
  --variants bias_relax_current,direct_qp_rank1_gated_q25_micro_adaptive50,direct_qp_curvature_gamma_kappa240 \
  --seeds 42 \
  --trials 500 \
  --steps-per-walk 8 \
  --device cuda

echo "completed $(date -Is)"
