#!/bin/bash
#SBATCH --job-name=ssw-lj-probe
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ssw-native-lj-panel-20261007/probe-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_panel=research/ga_ssw/evidence/ssw-native-lj-panel-20261007
task_prepared=$1
for task_n in 55 38; do
    /home/gengjianrui/.conda/envs/mace_env/bin/python "$task_panel/run_panel.py" \
        --execute --arm native --n "$task_n" --input "$task_prepared/lj${task_n}.extxyz" \
        --output "$task_panel/probe-${SLURM_JOB_ID}-lj${task_n}" --cap 120 --wall 120 --steps 1
done
