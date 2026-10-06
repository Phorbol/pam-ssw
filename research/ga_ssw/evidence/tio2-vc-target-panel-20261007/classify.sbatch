#!/usr/bin/env bash
#SBATCH --job-name=tio2-cold-phase
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-vc-target-panel-20261007/classify-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
task_dir=research/ga_ssw/evidence/tio2-vc-target-panel-20261007
/home/gengjianrui/.conda/envs/mace_env/bin/python "$task_dir/classify_cold.py" --panel "$task_dir/readout-1663551/analysis.json" --out "$task_dir/classify-${SLURM_JOB_ID}"
