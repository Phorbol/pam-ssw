#!/bin/bash
#SBATCH --job-name=native-cell-algebra
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/algebra-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/algebra-%j.err
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$1:/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/tools/unicorn-2.1.4"
output="research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/algebra-$SLURM_JOB_ID"
mkdir "$output"
cp research/ga_ssw/probe_native_cell_saved_mode.py "$output/runner.py"
exec /home/gengjianrui/.conda/envs/mace_env/bin/python "$output/runner.py" --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp > "$output/result.json"
