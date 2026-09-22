#!/bin/bash
#SBATCH --job-name=ga-row-order
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/ga-row-order-20260922/tests-%j.txt
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_ga_row_order.py tests/standalone/test_ga_checkpoint.py tests/standalone/test_paper_ga.py

/home/gengjianrui/.conda/envs/mace_env/bin/python -m research.ga_ssw.audit_legacy_permutation_invariance --manifest research/ga_ssw/evidence/s1-cutoff-contract-20260922/legal-permutation-manifest.json --output research/ga_ssw/evidence/ga-row-order-20260922/permutations-${SLURM_JOB_ID}.json
