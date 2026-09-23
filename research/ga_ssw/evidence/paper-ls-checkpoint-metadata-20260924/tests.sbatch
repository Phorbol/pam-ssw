#!/bin/bash
#SBATCH --job-name=paper-ls-cp-meta
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/paper-ls-checkpoint-metadata/research/ga_ssw/evidence/paper-ls-checkpoint-metadata-20260924/%x-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/paper-ls-checkpoint-metadata/research/ga_ssw/evidence/paper-ls-checkpoint-metadata-20260924/%x-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/gengjianrui/bin/pam-ssw-worktrees/paper-ls-checkpoint-metadata
cd /home/gengjianrui/bin/pam-ssw-worktrees/paper-ls-checkpoint-metadata
mode="$1"
python=/home/gengjianrui/.conda/envs/mace_env/bin/python
printf 'mode=%s host=%s branch=%s head=%s\n' "$mode" "$(hostname)" \
  "$(git branch --show-current)" "$(git rev-parse HEAD)"
if [[ "$mode" == red ]]; then
  "$python" -m pytest -q tests/standalone/test_ssw_checkpoint.py::test_fixed_cell_ls_checkpoint_resume_matches_continuous
elif [[ "$mode" == green ]]; then
  "$python" -m pytest -q tests/standalone/test_ssw_checkpoint.py \
    tests/standalone/test_recovered_rotation_ls.py tests/standalone/test_ls_pool_restart.py
  "$python" - <<'PY'
from pamssw.standalone.paper_reference import load_ssw_checkpoint
path = "/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c4h6-mh1-lifecycle-20260924/paper_ls/checkpoint.pkl"
checkpoint = load_ssw_checkpoint(path)
print({"legacy_checkpoint": path, "status": checkpoint.status,
       "next_index": checkpoint.next_index, "response_steps": checkpoint.response.steps,
       "response_last": checkpoint.response.last_response})
PY
else
  echo "expected mode red or green" >&2
  exit 2
fi
