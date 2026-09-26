#!/bin/bash
#SBATCH --job-name=c4h6-rearchive
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-3%4
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/pool-ase-rearchive-20260926/slurm-%A_%a.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/pool-ase-rearchive-20260926/slurm-%A_%a.err

set -euo pipefail
OWN_ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
MAIN_ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
PY=/home/gengjianrui/.conda/envs/mace_env/bin/python
case "$SLURM_ARRAY_TASK_ID" in
  0) ARM=ssw-seed61 ;;
  1) ARM=ssw-seed67 ;;
  2) ARM=native_ls-seed61 ;;
  3) ARM=native_ls-seed67 ;;
  *) exit 2 ;;
esac
cd "$MAIN_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$MAIN_ROOT:$OWN_ROOT/research/ga_ssw/evidence/ls-climb-depth-panel-20260926"
"$PY" "$OWN_ROOT/research/ga_ssw/evidence/pool-ase-rearchive-20260926/rearchive.py" \
  --main-root "$MAIN_ROOT" \
  --arm-index "$SLURM_ARRAY_TASK_ID" \
  --output "$OWN_ROOT/research/ga_ssw/evidence/pool-ase-rearchive-20260926/${ARM}.json"
