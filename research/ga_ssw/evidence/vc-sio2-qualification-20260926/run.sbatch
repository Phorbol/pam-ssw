#!/bin/bash
#SBATCH --job-name=vc-sio2-quench
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-sio2-qualification-20260926/slurm-%j.out
set -uo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
evidence=research/ga_ssw/evidence/vc-sio2-qualification-20260926
run="$evidence/run-${SLURM_JOB_ID}"
mkdir "$run"
cp "$evidence/qualification.json" "$run/qualification.json"
git rev-parse HEAD > "$run/git-head.txt"
timeout --signal=TERM --kill-after=5s 840s \
  /home/gengjianrui/.conda/envs/mace_env/bin/python \
  "$evidence/qualify.py" --plan "$run/qualification.json" \
  --out "$run/results" --execute > "$run/worker.stdout" 2>&1
rc=$?
printf '%s\n' "$rc" > "$run/worker.exit"
exit "$rc"
