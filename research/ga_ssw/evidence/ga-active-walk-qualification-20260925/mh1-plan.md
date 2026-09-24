# MH-1 C60 active fine-walk recovery probe

## Purpose and gate

This is a bounded execution and checkpoint qualification of GA active-walk
recovery across independent Python processes. It does not estimate GA search
quality, cage discovery, or general MACE trajectory reproducibility. The EMT
qualification in the parent `plan.md` must pass, including focused regression
and independent review, before this probe is run. EMT CPU1483815 has passed all four phases; the root agent has reviewed the result and this bounded follow-up.

The single comparison is one uninterrupted fixture against one pause/save and
fresh-process resume of the same frozen fixture. A successful result establishes
that the saved nested SSW continuation is loadable, its active phase and step
accounting are coherent, request accounting is conserved, and resumed output
does not re-ingest accepted landings. GPU trajectories may differ numerically;
no bitwise identity requirement is imposed.

## Frozen inputs and settings

- Use the first three raw C60 frames, in order, from
  `population-comparison-20260923/c60-seed3.json`:
  `c60-17093.extxyz`, `c60-17094.extxyz`, `c60-17095.extxyz`.
- Reuse that config's descriptor references derived from the original frames,
  descriptor bond lengths, weights and neighbor range; use its original MH-1
  model path, model hash, `omol` head, CUDA device and float64 dtype.
- Keep every scientific SSW and GA setting from the config, with only the
  engineering fixture lengths changed to `quick_steps=1`, `generations=0`,
  `fine_steps=2`, `cycles=1`. These values exercise a pause during the fine
  walk and are not scientific defaults. Set `NativeMCSettings` explicitly
  with `energy_tol=energy_tol_eV` and the configured `maxtrap`.
- Use one Torch thread, deterministic algorithms, TF32 disabled and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A resume process starts with a deliberately
  different fresh NumPy RNG; the checkpoint must replace it with saved state.
- Preserve each output in a new `mh1-runs/{full,split,resumed}` directory.
  Existing output causes an error; no overwrite, retry, or recovery script is
  provided. Save the trusted checkpoint, serialized result, summary, exact
  effective config, input/model hashes, runtime versions, core Git SHA/tree and
  worktree patch hash, and actual request counts.

## Bounded execution

Run these as three separate Python processes from this directory, within one
30-minute V100 allocation. Apply one 29-minute wall timeout to the complete
sequence, leaving one minute for job shutdown and output flushing:

```bash
timeout 29m bash -c 'python mh1_probe.py full && python mh1_probe.py pause && python mh1_probe.py resume && python mh1_probe.py analyze'
```

Each equivalent search trajectory is capped at 12,000 MACE surface requests:
the uninterrupted run may use at most 12,000, and the split prefix plus resume
suffix together may use at most 12,000. Thus the aggregate search ceiling is
24,000 requests. Offline force qualification is capped at eight fresh model
evaluations total (best and current geometry for full and resumed).
Allow at most one 30-minute V100 allocation for the three search invocations
and analysis combined; no automatic job submission is part of the runner.
There are no hidden retries. On a failed assertion or execution error, retain
all outputs and stop for review.

## Acceptance checks

1. The pause callback captures exactly one `active_walk.phase == 'fine'`
   checkpoint after a completed nested SSW outer step. Pickle roundtrip retains
   active phase, the two-step budget, nested SSW next-step position and request
   accounting. The checkpoint must demonstrate a nonzero completed-step count
   while work remains.
2. The resumed run's lineage request count equals saved prefix plus actual
   suffix requests and stays within 12,000. The uninterrupted count is reported
   independently; trajectory-dependent request counts need not match.
3. IDs are unique. For each completed walk, check the GA ingestion count
   against the SSW contract: every converged minimum appears in `minima`, and
   only nonconverged record landings add observations. Compare summed expected
   walk landings with quick/fine stage observation counts; retain observations
   and failures for review. On resume, verify the saved prefix observations are
   retained exactly once in their original order.
4. Fresh MH-1 force/energy checks run on the best archive geometry and
   `walks[-1].current` from both full and resumed results. Report measured
   energies and maximum forces; mark numerical qualification at the unchanged
   GA quench threshold `fmax <= 0.03 eV/A`. This check does not establish cage
   quality or global-search success.
5. Report numerical trajectory differences in request totals, stage outcomes,
   observation and accepted-landing counts, archive count, and fresh physical
   force results. Equality of serialized result objects is diagnostic only and
   is not an acceptance condition on CUDA.

The root agent reviews the artifacts and decides whether to authorize the
bounded run after the EMT gate. No code, model, scientific parameter, or
resource change is implied by this prepared probe.

Implementation provenance clarification (while GPU1483856 runs; no runtime change):
this fixture inherits the JSON configuration and uses the public default
`descriptor_row_order='legacy_counts'`. The older population-comparison runner
separately passes `full_fingerprint`; that argument is absent from its JSON.
Thus this is not a replay of that older population experiment. Full and split
arms here use the same legacy identity mode, appropriate for the checkpoint
contract under test; no cross-experiment search comparison is made. EMT uses
the same public default. This clarification does not change frozen code or
relax any recovery acceptance criterion.
