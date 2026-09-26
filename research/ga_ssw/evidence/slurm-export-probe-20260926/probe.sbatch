#!/bin/bash
#SBATCH --job-name=ssw-export-probe
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/optimizer-panel-20260926/research/ga_ssw/evidence/slurm-export-probe-20260926/probe-%j.out
set -eu
printf 'batch_body_started job=%s node=%s marker=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "${SSW_EXPORT_PROBE-unset}"
for key in SLURM_JOB_PARTITION SLURM_GPUS_ON_NODE SLURM_CPUS_ON_NODE SLURM_CPUS_PER_TASK SLURM_EXPORT_ENV; do
    printf '%s=%s\n' "$key" "${!key-unset}"
done
printf 'batch_body_completed\n'
