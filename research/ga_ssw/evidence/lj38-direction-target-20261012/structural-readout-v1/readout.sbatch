#!/bin/bash
#SBATCH --job-name=lj38-structure-readout
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012/research/ga_ssw/evidence/lj38-direction-target-20261012/structural-readout-v1/slurm-%j.out
set -euo pipefail

cd /home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_python=/home/gengjianrui/.conda/envs/mace_env/bin/python
task_evidence=research/ga_ssw/evidence/lj38-direction-target-20261012
task_output="$task_evidence/structural-readout-v1/output-${SLURM_JOB_ID}"
mkdir -p "$task_output"
exec "$task_python" "$task_evidence/structural-readout-v1/readout.py" \
  --run "$task_evidence/runs/run-1755511-0" \
  --run "$task_evidence/runs/run-1755511-1" \
  --output "$task_output"
