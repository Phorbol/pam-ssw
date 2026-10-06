#!/bin/bash
#SBATCH --job-name=lasp-seed-probe
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/lasp-explicit-seed-qualification-20261007/cpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/lasp-explicit-seed-qualification-20261007/cpu-%j.err
set -euo pipefail

repo=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
experiment_dir="$repo/research/ga_ssw/evidence/lasp-explicit-seed-qualification-20261007"
: "${SLURM_JOB_ID:?must run under Slurm}"
output="$experiment_dir/run-${SLURM_JOB_ID}"
socket_tmp="${HOME}/.rs${SLURM_JOB_ID}"
mkdir -m 700 "$socket_tmp"
export TMPDIR="$socket_tmp"
trap 'rm -rf -- "$socket_tmp"' EXIT

cd "$repo"
export PYTHONNOUSERSITE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

timeout --signal=TERM --kill-after=5s 260s \
    /home/gengjianrui/.conda/envs/mace_env/bin/python "$experiment_dir/probe.py" --execute "$output"
