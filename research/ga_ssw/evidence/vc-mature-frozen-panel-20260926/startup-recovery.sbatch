#!/bin/bash
#SBATCH --job-name=vc-startup-recovery
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-mature-frozen-panel-20260926/gpu-%j.out
set -uo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
panel=research/ga_ssw/evidence/vc-mature-frozen-panel-20260926
run="$panel/startup-recovery-$SLURM_JOB_ID"
mkdir "$run"
cp "$panel/plan.json" "$run/plan.json"
git rev-parse HEAD > "$run/git-head.txt"
# Only the task that exited before task.json/any PES; preserve the original run.
# Search deadline remains 100 seconds and search cap 600; allow import startup longer.
timeout --signal=TERM --kill-after=5s 360s \
  /home/gengjianrui/.conda/envs/mace_env/bin/python -m research.ga_ssw.run_vc_frozen_worker \
  --plan "$run/plan.json" --case-index 0 --method safe-lbfgs-total \
  --phase biased --out "$run/case0-safe-lbfgs-total-biased" > "$run/worker.stdout" 2>&1
worker_rc=$?
printf '%s\t%s\n' case0-safe-lbfgs-total-biased "$worker_rc" > "$run/worker-exits.tsv"
exit "$worker_rc"
