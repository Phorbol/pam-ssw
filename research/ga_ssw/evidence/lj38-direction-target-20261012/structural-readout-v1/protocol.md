# LJ38 completed-pair structural readout (v1)

This is a zero-PES post-hoc readout for completed slots 0 and 1 of Slurm array
1755511. It reads their immutable `progress.jsonl`, `outer-structures.extxyz`,
`search-ef-ledger.jsonl`, `result.json`, and `best.extxyz` artifacts and writes
only to a new job-specific directory under this readout folder. It does not
edit campaign files or either run directory.

For saved initial and outer minima that are converged with `fmax <= 0.05 eV/Å`
and single-component under the frozen 1.3 sigma graph, the script uses the
existing `analyze_panel.compare_geometry` graph mapping and proper-rotation
RMS test independently against the Cambridge global-minimum coordinates and
the previously qualified OPTIM second-lowest minimum. It records each first
positive match in cumulative paid search requests, then skips later comparisons
to that reference; this bounds work without changing the first-match result.
Mapping-cap outcomes remain `inconclusive`. It also compares the stored best
structure and includes the existing fresh best E/F record.

Structural identity is a separate post-hoc diagnostic. The predeclared target
remains the joint energy, force, connectivity, geometry, and fresh-validation
criterion; a structural match alone does not count as a joint target hit.
Execution status is preserved verbatim. In particular, `evaluation_failed`
is not interpreted as a calculator error: denied requests and charged failed
calculator requests are counted separately from their ledgers. Inputs that
were fragmented are excluded from the connected structural-minimum scan but
remain visible as rejected qualification records in the readout.

The second-minimum reference is the archived, independently qualified OPTIM
`optim-finish.extxyz` with energy -173.25237760119302 eV and its neighboring
`optim-qualification.json`. Matching that endpoint does not establish a path,
funnel residence, or global-search success. Each geometry comparison is capped
at the unchanged 1,000 graph mappings; a capped nonmatch is inconclusive.

The single-core Slurm task is capped at five minutes and writes
`output-${SLURM_JOB_ID}/readout.json`. To prepare the requested pair readout
after review:

```bash
sbatch research/ga_ssw/evidence/lj38-direction-target-20261012/structural-readout-v1/readout.sbatch
```

Equivalent direct command from repository root:

```bash
/home/gengjianrui/.conda/envs/mace_env/bin/python \
  research/ga_ssw/evidence/lj38-direction-target-20261012/structural-readout-v1/readout.py \
  --run research/ga_ssw/evidence/lj38-direction-target-20261012/runs/run-1755511-0 \
  --run research/ga_ssw/evidence/lj38-direction-target-20261012/runs/run-1755511-1 \
  --output research/ga_ssw/evidence/lj38-direction-target-20261012/structural-readout-v1/output-local
```

The direct command also performs zero PES requests; choose a new or empty
output path. The Slurm output ID prevents collisions. This readout describes
only this completed pair and cannot retroactively alter its execution status or
predeclared joint-hit count.
