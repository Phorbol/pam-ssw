#!/bin/bash
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --job-name=vc-firstpass-test
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-firstpass-observer-20260926/tests-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/research/test_vc_lbfgs_baseline_runner.py tests/research/test_vc_e2e_optimizer_panel.py
