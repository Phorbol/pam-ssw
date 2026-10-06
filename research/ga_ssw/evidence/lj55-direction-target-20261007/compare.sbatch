#!/bin/bash
#SBATCH --job-name=lj55-domain-comparison
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/lj55-direction-target-20261007/compare-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
task_dir=research/ga_ssw/evidence/lj55-direction-target-20261007
exec /home/gengjianrui/.conda/envs/mace_env/bin/python "$task_dir/compare_domain_revisions.py" --v1 "$task_dir/readout-1663155/analysis.json" --v2 "$task_dir/readout-1663186/analysis.json" --output "$task_dir/comparison-${SLURM_JOB_ID}"
