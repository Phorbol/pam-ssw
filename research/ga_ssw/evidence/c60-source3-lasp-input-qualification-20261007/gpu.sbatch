#!/bin/bash -l
#SBATCH --job-name=c60-src3-lasp-q
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-lasp-input-qualification-20261007/gpu-%j.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-lasp-input-qualification-20261007/gpu-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_DIR=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-lasp-input-qualification-20261007
# run_lasp uses tempfile.TemporaryDirectory for its Unix socket. Keep the path
# short and private so it stays below AF_UNIX's path limit and outside shared /tmp.
SOCKET_TMP="${HOME}/.c60q-${SLURM_JOB_ID}"
mkdir -m 700 "$SOCKET_TMP"
export TMPDIR="$SOCKET_TMP"
cleanup() { rm -rf -- "$SOCKET_TMP"; }
trap cleanup EXIT
/home/gengjianrui/.conda/envs/mace_env/bin/python "$TASK_DIR/qualify.py"
