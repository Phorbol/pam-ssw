# Positional-argument recovery of the frozen SiO2 panel

Code correction 52867a4 changes only SBATCH script argument transport. No
calculator, optimizer, input, tolerance or scientific budget changed. Shell
syntax and `git diff --check` passed. The CPU startup contrast is archived in
`../slurm-export-probe-20260926/`.

GPU 1502803 runs array index 0 (Safe-total, seed 71). A single startup check
at about 1.5 minutes verified RUNNING and actual worker configuration/input/
search-ledger artifacts, unlike the cancelled original attempts. This verifies
execution recovery, not numerical or physical qualification.

GPU 1502823 runs only indices 1–5 with `afterany:1502803`, at most two
concurrent GPUs. The first arm is not repeated. The two arrays together retain
the original six arms, 36000 search + 24 fresh maximum and 3 GPU-hour scientific
allocation ceiling; earlier cancelled allocations and two one-second CPU probes
remain separate execution overhead.

All submissions omit explicit `--export` and memory/CPU overrides. Commands:

```bash
sbatch --parsable --array=0 run.sbatch /absolute/path/to/prepared-plan-1501315/plan.json
sbatch --parsable --dependency=afterany:1502803 --array=1-5%2 run.sbatch /absolute/path/to/prepared-plan-1501315/plan.json
sbatch --parsable --dependency=afterany:1502823 readout.sbatch recovery-1502803
```

The exact full path is recorded in scheduler SubmitLine and each worker plan.
CPU 1502824 reads `arms-recovery-1502803/`: relative symlink `arm-0` points to
`../arms-1502803/arm-0`, and `arm-1` through `arm-5` point to the matching
`../arms-1502823/arm-*` directories. The existing merge/analyzer operates on
that derived view without copying raw outputs or making calculator calls.
Failure/censor accounting remains unchanged. Await numerical results before
making any scientific comparison.
