#!/bin/bash
#SBATCH --job-name=lj-vc-reference
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/lj-crystal-target-qualification-20261007/cpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/lj-crystal-target-qualification-20261007/cpu-%j.err
set -euo pipefail

repo=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
experiment_dir="$repo/research/ga_ssw/evidence/lj-crystal-target-qualification-20261007"
script="$experiment_dir/run.py"
if (($#)); then
    output=$1
else
    : "${SLURM_JOB_ID:?must run under Slurm}"
    output="$experiment_dir/run-${SLURM_JOB_ID}"
fi

cd "$repo"
export PYTHONNOUSERSITE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

exec timeout --signal=TERM --kill-after=10s 520s \
    /home/gengjianrui/.conda/envs/mace_env/bin/python "$script" --output "$output"
