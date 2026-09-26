#!/bin/bash
#SBATCH --job-name=matcher-qual
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/matcher-qualification-20260926/slurm-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/matcher-qualification-20260926/slurm-%j.err

set -euo pipefail
ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
MATCHER_ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
PY=/home/gengjianrui/.conda/envs/mace_env/bin/python
cd "$ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$MATCHER_ROOT"
"$PY" "$ROOT/research/ga_ssw/evidence/matcher-qualification-20260926/probe.py" \
  --audit "$ROOT/research/ga_ssw/evidence/ls-climb-depth-panel-20260926/topology-audit.json" \
  --matcher-root "$MATCHER_ROOT" \
  --output "$ROOT/research/ga_ssw/evidence/matcher-qualification-20260926/result.json"
