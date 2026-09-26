#!/bin/bash
#SBATCH --job-name=vc-sio2-heldout
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926/slurm-%A_%a.out
set -uo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PLAN_JSON=${1:?Pass the reviewed CPU-generated plan.json as the first argument}
if [[ ! -s "$PLAN_JSON" ]]; then
  echo "missing plan: $PLAN_JSON" >&2
  exit 2
fi
evidence=research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926
panel="$evidence/arms-$SLURM_ARRAY_JOB_ID"
run="$panel/arm-$SLURM_ARRAY_TASK_ID"
mkdir -p "$run"
cp "$PLAN_JSON" "$run/plan.json"
sha256sum "$PLAN_JSON" > "$run/plan.sha256"
git rev-parse HEAD > "$run/git-head.txt"
methods=(safe_total ase scipy)
seeds=(71 83)
method=${methods[$((SLURM_ARRAY_TASK_ID / 2))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 2))]}
name="case0-${method}-seed${seed}"
timeout --signal=TERM --kill-after=5s 1740 \
  /home/gengjianrui/.conda/envs/mace_env/bin/python -m research.ga_ssw.run_vc_e2e_optimizer_panel \
  --plan "$run/plan.json" --case-index 0 --method "$method" --seed "$seed" \
  --out "$run/$name" > "$run/$name.stdout" 2>&1
rc=$?
printf '%s\t%s\n' "$name" "$rc" > "$run/worker-exits.tsv"
exit "$rc"
