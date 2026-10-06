#!/bin/bash
#SBATCH --job-name=ssw-lj-same-pes
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:12:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ssw-native-lj-panel-20261007/panel-%A_%a.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_python=/home/gengjianrui/.conda/envs/mace_env/bin/python
task_panel=research/ga_ssw/evidence/ssw-native-lj-panel-20261007
task_prepared=$1
task_arms=(rotation full native)
task_sizes=(55 38)
task_arm=${task_arms[$((SLURM_ARRAY_TASK_ID % 3))]}
task_n=${task_sizes[$((SLURM_ARRAY_TASK_ID / 3))]}
exec "$task_python" "$task_panel/run_panel.py" --execute --arm "$task_arm" --n "$task_n" \
    --input "$task_prepared/lj${task_n}.extxyz" \
    --output "$task_panel/run-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}" --cap 4000 --wall 600 --steps 6
