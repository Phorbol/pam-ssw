#!/bin/bash
#SBATCH --job-name=ssw-height-only
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
#SBATCH --output=research/ga_ssw/evidence/lj-gaussian-policy-probe-20260927/cpu-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONPATH=.
exec /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/run_lj_gaussian_policy_probe.py --execute --policies pam_curvature_height_only --output "research/ga_ssw/evidence/lj-gaussian-policy-probe-20260927/run-${SLURM_JOB_ID}"
