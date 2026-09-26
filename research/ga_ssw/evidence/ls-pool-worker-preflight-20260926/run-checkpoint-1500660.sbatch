#!/bin/bash
#SBATCH --job-name=ls-pool-emt
#SBATCH --partition=CPU-MISC
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/ls-pool-worker-preflight-20260926/slurm-checkpoint-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MAINLINE=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
LSCHECK=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
PLAN="$MAINLINE/research/ga_ssw/evidence/ls-pool-routing-20260926/plan.json"
WORKER="$PWD/research/ga_ssw/run_ls_pool_routing_panel.py"
BASE="$PWD/research/ga_ssw/evidence/ls-pool-worker-preflight-20260926/outputs-checkpoint-1500660"
export PYTHONPATH="$MAINLINE" PAMSSW_MAINLINE="$MAINLINE"
PYTHON=/home/gengjianrui/.conda/envs/mace_env/bin/python
cd "$LSCHECK"
PYTHONPATH="$LSCHECK:$MAINLINE" "$PYTHON" -m pytest -q tests/research/test_ls_pool_routing_analysis.py \
  > /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/ls-pool-worker-preflight-20260926/analyzer-pytest.txt 2>&1
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$MAINLINE" PAMSSW_MAINLINE="$MAINLINE"
"$PYTHON" "$WORKER" --plan "$PLAN" --case-index 0 --mode mc --emt-preflight --out "$BASE/mc"
"$PYTHON" "$WORKER" --plan "$PLAN" --case-index 0 --mode uniform --emt-preflight --out "$BASE/uniform"
"$PYTHON" "$WORKER" --plan "$PLAN" --case-index 0 --mode pam --emt-preflight --out "$BASE/pam"
"$PYTHON" research/ga_ssw/evidence/ls-pool-worker-preflight-20260926/verify.py "$BASE"
