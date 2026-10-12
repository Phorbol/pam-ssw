#!/bin/bash
#SBATCH --job-name=native-ext-stress
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=DSPRHBM
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/native-external-stress-20261012/research/ga_ssw/evidence/native-external-stress-20261012/cpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/native-external-stress-20261012/research/ga_ssw/evidence/native-external-stress-20261012/cpu-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKTREE=/home/gengjianrui/bin/pam-ssw-worktrees/native-external-stress-20261012
EVIDENCE="$WORKTREE/research/ga_ssw/evidence/native-external-stress-20261012"
RUN="$EVIDENCE/run-${SLURM_JOB_ID}"
cd "$WORKTREE"
/home/gengjianrui/.conda/envs/mace_env/bin/python "$EVIDENCE/run.py" --prepare "$RUN"
/home/gengjianrui/.conda/envs/mace_env/bin/python "$EVIDENCE/run.py" --execute "$RUN"
