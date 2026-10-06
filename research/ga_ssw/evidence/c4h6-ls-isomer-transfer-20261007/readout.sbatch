#!/bin/bash
#SBATCH --job-name=c4h6-ls-readout
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c4h6-ls-isomer-transfer-20261007/readout-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_dir=research/ga_ssw/evidence/c4h6-ls-isomer-transfer-20261007
exec /home/gengjianrui/.conda/envs/mace_env/bin/python "$task_dir/analyze.py" --run "$task_dir/run-1662702" --output "$task_dir/readout-${SLURM_JOB_ID}"
