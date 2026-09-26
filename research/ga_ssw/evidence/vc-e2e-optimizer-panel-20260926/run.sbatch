#!/bin/bash
#SBATCH --job-name=vc-e2e-optimizers
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-e2e-optimizer-panel-20260926/gpu-%A_%a.out
set -uo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
panel=research/ga_ssw/evidence/vc-e2e-optimizer-panel-20260926
run="$panel/run-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir "$run"
cp "$panel/plan.json" "$run/plan.json"
git rev-parse HEAD > "$run/git-head.txt"
for seed in 71 83; do
  for method in safe_total ase scipy; do
    name="case${SLURM_ARRAY_TASK_ID}-${method}-seed${seed}"
    timeout --signal=TERM --kill-after=5s 360s \
      /home/gengjianrui/.conda/envs/mace_env/bin/python -m research.ga_ssw.run_vc_e2e_optimizer_panel \
      --plan "$run/plan.json" --case-index "$SLURM_ARRAY_TASK_ID" --method "$method" \
      --seed "$seed" --out "$run/$name" > "$run/$name.stdout" 2>&1
    worker_rc=$?
    printf '%s\t%s\n' "$name" "$worker_rc" >> "$run/worker-exits.tsv"
  done
done
