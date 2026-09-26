#!/bin/bash
#SBATCH --job-name=ls-pool-readout-check
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:02:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-pool-routing-20260926/analysis-preflight/slurm-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926:/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
cd /home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/research/test_ls_pool_routing_analysis.py
/home/gengjianrui/.conda/envs/mace_env/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
from research.ga_ssw.analyze_ls_pool_routing_panel import load_arm, shared_prefix
p=Path('research/ga_ssw/analyze_ls_pool_routing_panel.py')
print('analyzer_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
base=Path('/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/ls-pool-worker-preflight-20260926/outputs-checkpoint-1500660')
rows=[]
for mode in ['mc','uniform','pam']:
 plan=json.loads((base/mode/'effective-plan.json').read_text())
 row=load_arm(base/mode,plan['cases'][0],mode,plan)
 assert not row['issues'],(mode,row['issues'])
 rows.append(row)
 print(mode,row['status'],row['search_requests'],row['fresh_certified'])
assert shared_prefix(rows,['mc','uniform','pam'])==64
print('REAL_EMT_ARTIFACT_ACCOUNTING_PASS shared_prefix=64')
PY
