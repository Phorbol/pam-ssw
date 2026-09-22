#!/bin/bash
#SBATCH --job-name=ls-pool-restart
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/ls-pool-restart-20260922/gpu-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_exec=/home/gengjianrui/.conda/envs/mace_env/bin/python
run_dir=research/ga_ssw/evidence/ls-pool-restart-20260922/prepared-v2
exec "$python_exec" "$run_dir/runner.py" --execute --output "$run_dir"
