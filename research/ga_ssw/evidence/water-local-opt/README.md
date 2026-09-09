# Original LASP local-optimization evidence

This is a review snapshot, not a standalone binary/potential distribution.
The original run directory is:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/runs/water-local-opt`.

`run_bounded.py` records the script executed **from that original directory**;
its relative executable path is not valid from this snapshot. Input structure,
potential and untouched logs remain there. To repeat in a separate run directory,
copy the same supplied inputs, use the recorded `lasp.in`, load
`intel/mpi/2021.13`, and run the script with its executable path adjusted to the
original binary and OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1.
Do not overwrite the recorded run.

Trailing whitespace was removed from the review copies of `lasp.in`, `lasp.out`
and `allkeys.log`; their numerical content is unchanged. Source file SHA256 values
are in `provenance.json`. `run-status.json` records execution, not independent
force certification. This was one local optimization, not a full GA-SSW search.
