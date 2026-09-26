#!/bin/bash
#SBATCH --job-name=vc-sio2-plan
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926/prepare-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
evidence=research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926
out="$evidence/prepared-plan-$SLURM_JOB_ID"
mkdir "$out"
git rev-parse HEAD > "$out/git-head.txt"
/home/gengjianrui/.conda/envs/mace_env/bin/python \
  "$evidence/build_plan.py" --out "$out/plan.json" > "$out/build-summary.json"
sha256sum "$evidence/build_plan.py" "$out/plan.json" > "$out/sha256sums.txt"
