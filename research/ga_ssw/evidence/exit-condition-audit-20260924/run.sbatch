#!/bin/bash
#SBATCH --job-name=ssw-exit-audit
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=research/ga_ssw/evidence/exit-condition-audit-20260924/cpu-%j.out
#SBATCH --error=research/ga_ssw/evidence/exit-condition-audit-20260924/cpu-%j.err
set -eu
python research/ga_ssw/evidence/exit-condition-audit-20260924/analyze.py
