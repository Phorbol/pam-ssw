#!/bin/bash
#SBATCH --job-name=ssw-lj-readout
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ssw-native-lj-panel-20261007/final-readout-%j.out
set -euo pipefail

task_checkout=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
task_panel="$task_checkout/research/ga_ssw/evidence/ssw-native-lj-panel-20261007"
task_python=/home/gengjianrui/.conda/envs/mace_env/bin/python
task_prepared="$task_panel/prepared-1662572"
task_output="$task_panel/analysis-final-readout-${SLURM_JOB_ID}"

cd "$task_checkout"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Fixed slot order: LJ55 rotation/full/native, then LJ38 rotation/full/native.
# Offline analysis only. Deliberately omit --fresh-native (no new E/F calls).
exec "$task_python" "$task_panel/analyze_panel.py" \
    --prepared "$task_prepared" \
    --runs \
    "$task_panel/readout-1662743/python-0" \
    "$task_panel/run-1662763-1" \
    "$task_panel/run-1662691-2" \
    "$task_panel/readout-1662743/python-3" \
    "$task_panel/run-1662763-4" \
    "$task_panel/run-1662609-5" \
    --output "$task_output"
