#!/bin/bash
#SBATCH --job-name=lj38-second-structure
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012/research/ga_ssw/evidence/lj38-direction-target-20261012/structural-readout-v1/second-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_dir=research/ga_ssw/evidence/lj38-direction-target-20261012
exec /home/gengjianrui/.conda/envs/mace_env/bin/python "$task_dir/structural-readout-v1/readout.py" --run "$task_dir/runs/run-1755511-2" --run "$task_dir/runs/run-1755511-3" --output "$task_dir/structural-readout-v1/output-$SLURM_JOB_ID"
