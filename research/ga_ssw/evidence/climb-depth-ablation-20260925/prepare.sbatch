#!/bin/bash
#SBATCH --job-name=climb-depth-inputs
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=research/ga_ssw/evidence/climb-depth-ablation-20260925/cpu-%j.out
#SBATCH --error=research/ga_ssw/evidence/climb-depth-ablation-20260925/cpu-%j.err
set -eu
python research/ga_ssw/evidence/climb-depth-ablation-20260925/run.py --prepare
