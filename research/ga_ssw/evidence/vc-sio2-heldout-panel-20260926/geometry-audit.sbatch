#!/bin/bash
#SBATCH --job-name=vc-sio2-geom
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926/geometry-audit-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926/geometry-audit-$SLURM_JOB_ID/geometry.json
mkdir -p "$(dirname "$out")"
/home/gengjianrui/.conda/envs/mace_env/bin/python \
  research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926/geometry_audit.py \
  research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926 "$out"
sha256sum research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926/geometry_audit.py "$out" \
  > "$(dirname "$out")/sha256sums.txt"
