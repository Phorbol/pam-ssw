#!/bin/bash
#SBATCH --job-name=ssw-lj-readout
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ssw-native-lj-panel-20261007/readout-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_python=/home/gengjianrui/.conda/envs/mace_env/bin/python
task_dir=research/ga_ssw/evidence/ssw-native-lj-panel-20261007
task_output="$task_dir/readout-${SLURM_JOB_ID}"
mkdir "$task_output"
for task_id in 0 1 3 4; do
    "$task_python" "$task_dir/recover_python_readout.py" --raw "$task_dir/run-1662691-${task_id}" --output "$task_output/python-${task_id}" --fresh
done
exec "$task_python" "$task_dir/analyze_panel.py" --prepared "$task_dir/prepared-1662572" --runs \
 "$task_output/python-0" "$task_output/python-1" "$task_dir/run-1662691-2" \
 "$task_output/python-3" "$task_output/python-4" "$task_dir/run-1662609-5" \
 --output "$task_output/analysis" --fresh-native
