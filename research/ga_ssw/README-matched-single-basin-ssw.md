# Matched-cost single-basin SSW control

This is an executed, reproducible control for `fixed-ga-single-basin-20260912`.  It
starts from the same single G2 bicyclobutane structure, uses the same frozen
`pamssw` source snapshot, `SSWConfig`, GFN2-xTB settings and seeds 11/29, and
does not use GA proposal, archive, references or LS.  The SSW step limit is 100
as a defensive lifecycle bound.

The search caps are the realized GA search-request counts (12,000 for seed 11
and 11,154 for seed 29), so this is a matched-realized-cost control.  A search
request is written before/after every calculator call, including failures and
cap denials.  Returned minima are independently evaluated with a fresh GFN2
calculator; fresh requests and exceptions are retained in the per-seed
summary.  Search and fresh costs are reported separately.

The runner creates its output and frozen source snapshot during preparation;
`--execute` is required to run PES calls.  Example after review:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/pam-ssw-tblite-20260909:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/run_fixed_ssw_single_basin_matched.py \
  --output research/ga_ssw/evidence/NEW-matched-single-basin-ssw --execute
```

This is a lifecycle/cost control, not an independent efficiency benchmark.
Two seeds and one molecular basin cannot establish a method advantage.

Executed evidence: `evidence/fixed-ssw-single-basin-matched-20260912/`.
Both controls reached their paired request caps; 23154 search + 26 fresh
requests, with all 26 fresh force checks passing. Seed11 GA gained another
0.1433276 eV compared with SSW; seed29 differs by only 0.0000410 eV in the
opposite direction. Full audit: `root-final-audit.json` in that directory.
The example above requires a new output directory; do not overwrite archived runs.
