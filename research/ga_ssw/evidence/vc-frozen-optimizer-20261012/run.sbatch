#!/usr/bin/env bash
# Submission is intentionally not performed by this research runner task.
# One V100, one 30-minute wall allocation; omit cpus-per-task for GPU-derived 8 CPUs.
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

evidence_dir="/home/gengjianrui/bin/pam-ssw-worktrees/vc-optimizer-frozen-20261012/research/ga_ssw/evidence/vc-frozen-optimizer-20261012"
cd "$evidence_dir"
/home/gengjianrui/.conda/envs/mace_env/bin/python run.py --run --out "run-${SLURM_JOB_ID}"
