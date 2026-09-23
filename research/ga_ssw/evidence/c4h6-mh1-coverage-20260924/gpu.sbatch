#!/bin/bash
#SBATCH --job-name=c4h6-coverage
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/gpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/gpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c4h6-mh1-coverage-20260924
arms=(ssw paper_ls native_ls)
seeds=(61 67)
task_index=${SLURM_ARRAY_TASK_ID:?array task ID required}
exec /home/gengjianrui/.conda/envs/mace_env/bin/python run.py --arm "${arms[$((task_index / 2))]}" --seed "${seeds[$((task_index % 2))]}" --execute
