#!/bin/bash
#SBATCH --job-name=ls-mc-save-check
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:03:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/ls-pool-mc-postprocess-20260926/check-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export PAMSSW_MAINLINE="$PYTHONPATH"
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
python_bin=/home/gengjianrui/.conda/envs/mace_env/bin/python
"$python_bin" research/ga_ssw/run_ls_pool_routing_panel.py --emt-preflight --mode mc --case-index 0 \
 --plan "$PYTHONPATH/research/ga_ssw/evidence/ls-pool-routing-20260926/plan.json" \
 --out research/ga_ssw/evidence/ls-pool-mc-postprocess-20260926/run
"$python_bin" - <<'PY'
import json
from pathlib import Path
root=Path('research/ga_ssw/evidence')
old=root/'ls-pool-worker-preflight-20260926/outputs-checkpoint-1500660/mc'
new=root/'ls-pool-mc-postprocess-20260926/run'
s=json.loads((new/'summary.json').read_text())
assert s['status']=='completed'
assert (s['search_requests'],s['fresh_requests'])==(67,3)
assert not (new/'offline-identity.json').exists()
assert json.loads((new/'search-result.json').read_text())['result']==json.loads((old/'search-result.json').read_text())['result']
assert json.loads((new/'fresh-checks.json').read_text())==json.loads((old/'fresh-checks.json').read_text())
assert (new/'checkpoint.pkl').is_file() and (new/'minima.extxyz').is_file()
print('PASS: unchanged search and fresh results; summary/checkpoint/minima saved without MC offline matching; 67 search+3 fresh EMT calls')
PY
