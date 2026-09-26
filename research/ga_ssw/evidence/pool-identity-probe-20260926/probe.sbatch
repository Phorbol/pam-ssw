#!/bin/bash
#SBATCH --job-name=c4h6-pool-identity
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/pool-identity-probe-20260926/slurm-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926/research/ga_ssw/evidence/pool-identity-probe-20260926/slurm-%j.err
set -euo pipefail
matcher_root=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
agent_root=/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926
export PYTHONNOUSERSITE=1 PYTHONPATH="$matcher_root" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python \
  "$agent_root/research/ga_ssw/evidence/pool-identity-probe-20260926/probe.py" \
  --audit "$agent_root/research/ga_ssw/evidence/ls-climb-depth-panel-20260926/topology-audit.json" \
  --matcher-source-root "$matcher_root" \
  --output "$agent_root/research/ga_ssw/evidence/pool-identity-probe-20260926/result.json"
