#!/bin/bash
#SBATCH --job-name=ssw-contracts
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/s1-cutoff-contract-20260922/contracts-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity
export PYTHONNOUSERSITE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_exec=/home/gengjianrui/.conda/envs/mace_env/bin/python
result_dir=research/ga_ssw/evidence/s1-cutoff-contract-20260922
status=0
"$python_exec" -m pytest -q research/ga_ssw/test_s1_reference.py \
 tests/standalone/test_native_ls_driver.py::test_native_ls_update_uses_selected_current_after_mc_rejection \
 > "$result_dir/tests-${SLURM_JOB_ID}.txt" 2>&1 || status=$?
"$python_exec" -m research.ga_ssw.audit_legacy_permutation_invariance \
 --manifest research/ga_ssw/legacy_permutation_manifest.json \
 --output "$result_dir/permutation-${SLURM_JOB_ID}.json" || status=$?
exit "$status"
