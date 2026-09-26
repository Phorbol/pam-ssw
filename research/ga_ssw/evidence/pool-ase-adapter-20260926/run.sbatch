#!/bin/bash
#SBATCH --job-name=pool-ase-qual
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/pool-ase-adapter-20260926/slurm-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/pool-ase-adapter-20260926/slurm-%j.err

set -euo pipefail
OWN_ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
MAIN_ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
PY=/home/gengjianrui/.conda/envs/mace_env/bin/python
cd "$MAIN_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$MAIN_ROOT"
"$PY" "$OWN_ROOT/research/ga_ssw/evidence/pool-ase-adapter-20260926/probe.py" \
  --main-root "$MAIN_ROOT" \
  --output "$OWN_ROOT/research/ga_ssw/evidence/pool-ase-adapter-20260926/result.json"
