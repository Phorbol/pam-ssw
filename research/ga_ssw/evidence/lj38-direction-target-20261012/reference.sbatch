#!/bin/bash
#SBATCH --job-name=lj38-reference-ef
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:02:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012/research/ga_ssw/evidence/lj38-direction-target-20261012/reference-%j.out
set -euo pipefail

cd /home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_python=/home/gengjianrui/.conda/envs/mace_env/bin/python
task_evidence=research/ga_ssw/evidence/lj38-direction-target-20261012
mkdir -p "$task_evidence/runs"
task_output="$task_evidence/runs/reference-${SLURM_JOB_ID}.json"
"$task_python" "$task_evidence/run.py" --qualify-reference \
    --output "$task_output"
echo "reference_qualification=$task_output"
