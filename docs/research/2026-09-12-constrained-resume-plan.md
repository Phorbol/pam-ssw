# Constrained SSW attempt-boundary resume plan (2026-09-12)

The constrained attempt-boundary checkpoint is now implemented in the
fixed-substrate `run_constrained_ssw` entry point. The plan below records the
implemented contract and remaining real-system validation; it does not claim
that all constrained LS variants or calculators have been validated. The
active Cartesian chart and active-force certificate remain the source of truth.
whose active Cartesian chart and active-force certificate must remain the
source of truth.

## Contract

Resume is an explicit operation. A caller loads a trusted local checkpoint,
supplies a new physical surface/calculator, and requests additional outer
attempts:

```python
cp = load_constrained_checkpoint(path)
result = run_constrained_ssw(
    atoms, fresh_surface, steps=1, config=config, rng=np.random.default_rng(),
    fixed_indices=cp.fixed_indices,
    direction_fixed_indices=cp.direction_fixed_indices,
    ls=cp.ls_settings,
    checkpoint=cp,
    checkpoint_path=path,
)
```

`steps` always means new attempted outer iterations; an iteration may end in a
recoverable failed climb. A path argument only writes
the new boundary; an existing file never changes a new run into a resume.
Existing `run_rc_ssw` callers retain their current signature and lifecycle.
The constrained checkpoint has its own type because RC and constrained result
objects have different coordinate shapes and certificate semantics; it must
not be forced into `SSWCheckpoint`.

## Boundary and payload

Only a completed outer-attempt boundary is resumable. The payload should have
a schema/version and contain:

- calculator-free `initial`, `current`, and `best` constrained quench results,
  including their atoms, energies, active/full raw force certificates,
  optimizer summaries, and request deltas;
- all raw `minima` and `records`, including rejected and failed observations,
  their outer indices, landing/preparation records, active certificates,
  Gaussian centers/directions/weights, and per-attempt request costs;
- `next_index`, accumulated `evaluation_requests`, and the physical surface
  request count at the boundary, with no inferred cost from record length;
- a calculator-free chart reference (the fixed-atom reference positions and
  cell), `fixed_indices`, `direction_fixed_indices`, active-index ordering,
  coordinate label, and any constrained certificate scope;
- the complete `ConstrainedSSWConfig` and the LS selection. For paper LS,
  serialize settings, frozen bond/softening data, reference atoms, and
  response state, while omitting `physical_surface`, `soft_surface`, and every
  calculator. On restore, rebuild those links against the caller's fresh
  physical surface. Native LS is not part of the first constrained milestone;
  its separate runtime state must be specified before it is accepted here;
- the NumPy Generator `bit_generator.state`, preserving its bit-generator
  class and rejecting a mismatched class before any surface call;
- optional identity-view data only after constrained identity support is
  explicitly added. A callable matcher is never serialized.

All nested ASE objects must be copied with calculators removed. The save path
should use the existing trusted-pickle policy, write to a same-directory
temporary file, flush and replace atomically. Loading validates type, schema,
contiguous indices, request accounting, finite arrays, composition, PBC, cell,
masses, fixed masks, and config/LS signatures before constructing a surface
or making an E/F request.

## Failure semantics

An initial-quench failure, LS initialization failure, LS prequench failure, or
LS update failure is an early terminal diagnostic checkpoint after recording
the failed attempt's requests. Its `status` is terminal and resume must reject
it. Rotation, biased-quench, and true-quench failures retain the existing
`_run_reduced_ssw` control flow: they are recorded as failed attempts and the
overall `completed_with_failures` boundary remains resumable. In particular,
an LS prequench exception may occur after physical requests and that full cost
must remain in the record and checkpoint.

A hard kill inside an unfinished Gaussian or optimizer stage has no atomic
boundary. The interface cannot reconstruct it; an external calculator ledger
may account for paid work. `steps=0` on a new run should create the initial
boundary when checkpointing is enabled. `steps=0` on a valid completed
checkpoint should preserve it without another initial quench or surface call.

## Minimal implementation order

1. Calculator-free recursive copying, validation, atomic trusted-pickle save/load,
   and the constrained payload/result adapter are implemented without changing
   RC result shapes.
2. Restored `initial/current/best/minima/records`, `next_index`, RNG, chart
   masks/reference, and paper LS runtime are threaded through `_run_reduced_ssw`.
   The default path remains free of snapshots and existing RC calls are unchanged.
3. Add identity-view persistence only when a constrained caller API exists;
   do not pickle a matcher or silently rematch old observations.
5. Specify and implement native LS serialization separately, then add its
   wrapper forwarding. Do not represent native state as paper LS response
   state.

## Validation matrix

The minimum evidence should use the same source atoms, seed, config, and
calculator settings for each pair:

- Cu fixed-substrate EMT: two outer attempts continuously versus one attempt
  plus one resumed attempt; compare the full first-attempt ledger prefix,
  first resumed chart/current geometry, raw E/F calls, indices, RNG state,
  records, and accumulated request totals;
- Al fixed-substrate EMT: the same continuous/split replay to test a second
  composition and a different fixed/active mask;
- one paper-LS constrained smoke case: compare frozen softening, response
  state, table/bond data, and request ledger across the boundary;
- budget checks: initial plus per-record costs equal the physical ledger,
  resumed total equals prior plus the new delta, and no failed attempt cost is
  discarded;
- negative checks with a zero-call counting surface: changed composition,
  cell/PBC/masses, masks, config, LS settings, RNG class, schema, index, or
  cost must reject before the first E/F request; terminal checkpoints must
  reject as non-resumable.

Implemented checks currently cover Harmonic fixed-substrate replay, three-fixed
atom preflight, terminal diagnostics, and a paper-LS frozen-state interface
smoke. The Cu/Al EMT continuation matrix and fuller paper-LS ledger comparison
remain the forthcoming real-system evidence; no results are asserted here.
These tests establish state and accounting preservation. They do not establish
constrained search quality, basin identity, or physical validity beyond the
existing active-force certificate.

Cu/Al EMT matrix now completed: see 2026-09-12-constrained-resume-audit.md.
Cu exact replay; Al request totals agree but trajectory replay is not exact.
The broader paper-LS constrained real-system ledger comparison remains pending.
