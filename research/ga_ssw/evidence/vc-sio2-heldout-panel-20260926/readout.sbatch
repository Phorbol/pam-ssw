#!/bin/bash
#SBATCH --job-name=vc-sio2-readout
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926/readout-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PANEL_JOB=${1:?Pass the GPU array job ID as the first argument}
evidence=research/ga_ssw/evidence/vc-sio2-heldout-panel-20260926
view="$evidence/analysis-view-$SLURM_JOB_ID"
py=/home/gengjianrui/.conda/envs/mace_env/bin/python
"$py" "$evidence/merge_analysis_view.py" --arms "$evidence/arms-$PANEL_JOB" --out "$view"
"$py" research/ga_ssw/evidence/vc-e2e-optimizer-panel-20260926/analyze.py \
  "$view/run-0" --output "$evidence/readout-$SLURM_JOB_ID"
