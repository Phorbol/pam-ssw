#!/bin/bash
#SBATCH --job-name=native-vc-trace
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/cpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/cpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$1"
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/probe_native_vc_lifecycle.py "$1/research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/run-$SLURM_JOB_ID" --execute
