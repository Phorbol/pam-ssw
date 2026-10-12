#!/bin/bash
#SBATCH --job-name=lj38-direction-smoke
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:04:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012/research/ga_ssw/evidence/lj38-direction-target-20261012/smoke-%A_%a.out
set -euo pipefail

cd /home/gengjianrui/bin/pam-ssw-worktrees/benchmark-lj38-20261012
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_python=/home/gengjianrui/.conda/envs/mace_env/bin/python
task_evidence=research/ga_ssw/evidence/lj38-direction-target-20261012
task_reference=${1:?pass the successful runs/reference-JOBID.json from reference.sbatch}
if [[ ! -s "$task_reference" ]]; then
    echo "missing independent LJ38 reference qualification: $task_reference" >&2
    exit 2
fi
mkdir -p "$task_evidence/runs"
task_output="$task_evidence/runs/smoke-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
exec "$task_python" "$task_evidence/run.py" --smoke \
    --slot "$SLURM_ARRAY_TASK_ID" --output "$task_output" \
    --reference-qualification "$task_reference"
