#!/bin/bash
#SBATCH --job-name=vc-frozen-optimizers
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-mature-frozen-panel-20260926/gpu-%j.out
set -uo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
panel=research/ga_ssw/evidence/vc-mature-frozen-panel-20260926
run="$panel/run-$SLURM_JOB_ID"
mkdir "$run"
cp "$panel/plan.json" "$run/plan.json"
git rev-parse HEAD > "$run/git-head.txt"
for case_index in 0 1; do
  for method in safe-lbfgs-total ase-lbfgs-linesearch scipy-lbfgsb; do
    for phase in biased unbiased; do
      name="case${case_index}-${method}-${phase}"
      timeout --signal=TERM --kill-after=5s 120s \
        /home/gengjianrui/.conda/envs/mace_env/bin/python -m research.ga_ssw.run_vc_frozen_worker \
        --plan "$run/plan.json" --case-index "$case_index" --method "$method" \
        --phase "$phase" --out "$run/$name" > "$run/$name.stdout" 2>&1
      worker_rc=$?
      printf '%s\t%s\n' "$name" "$worker_rc" >> "$run/worker-exits.tsv"
    done
  done
done
