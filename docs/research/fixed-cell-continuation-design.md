# Fixed-cell SSW / LS outer-attempt continuation design

Status: bounded design review only; no implementation or new PES calculation.
The proposed boundary is between completed outer attempts in
`standalone.paper_reference.run_ssw`. It is not a Gaussian-internal restart
and it does not alter the direction, Gaussian, quench, MC or LS kernels.

## Current limitation

`run_ssw` always performs an initial true quench and initializes `current`,
`best`, `minima`, the optional frozen LS potential and the response controller
(`paper_reference.py:215-247`). A second call therefore pays the initial quench
again and cannot continue the same RNG/LS trajectory. The existing
`AtomicClimbCheckpoint` is a Gaussian-boundary substage checkpoint only; it
does not contain outer MC state or the LS response state
(`standalone/atomic_climb.py:16-44`, `:72-105`).

## Proposed boundary and observable

Add an explicit continuation API conceptually equivalent to:

```text
run_ssw(..., checkpoint=None, max_attempts=K)
```

The first call performs the existing initial quench and starts at outer attempt
0. It returns the normal `SSWResult` plus an optional continuation checkpoint
when it stops at a completed-attempt boundary. A resumed call consumes that
checkpoint, skips initial quench and begins at the next outer attempt. The
existing `steps` argument remains the total number of attempts requested for
that call, or is replaced by an explicitly documented `max_attempts`; this
choice must be fixed before implementation so a split run and one-shot run use
the same attempt indices.

The checkpoint boundary is valid only after the complete per-attempt sequence:

```text
LS prequench (if enabled) -> direction/Gaussian climb -> true landing quench
-> MC decision -> LS response/frozen-potential update -> record append
```

An interruption during rotation, Gaussian height preparation, biased quench,
true landing or LS update must be represented as a terminal partial record and
must not be advertised as resumable outer-boundary state. Resuming such a
partial attempt would require preserving pending Gaussian terms, work atoms,
optimizer state and exact failure semantics, which is a different design.

## Minimal explicit JSON schema

The file must be a versioned JSON object. No pickle, executable class path or
arbitrary object deserialization is allowed.

| Field | Required contents | Purpose / boundary |
|---|---|---|
| `schema` | fixed string and integer version | Reject incompatible formats explicitly. |
| `source` | package version/commit, module contract name | Provenance; never import code from the file. |
| `geometry` | atomic numbers, Cartesian positions, 3x3 cell, PBC flags, `cluster_frame`, `direction_sampling` | Reconstruct the exact fixed-cell state and validate identity/order/cell. Constraints are currently unsupported and must be rejected rather than serialized ambiguously. |
| `surface_contract` | calculator/backend name supplied by caller, model identifier/checksum if available, units, fixed-cell flag | The JSON cannot serialize a live calculator. Resume must require an explicitly supplied compatible surface; exact compatibility is otherwise `unknown`. |
| `current` | Atoms payload, true energy, max force, converged/surface certificate, source attempt index | Next accepted seed and MC reference. The energy and certificate are records, not permission to claim fresh physical validation. |
| `best` | Atoms payload, true energy, max force, converged/surface certificate | Preserve one-shot best result. |
| `history` | completed `SSWStep` records or a lossless JSON record encoding, minima references/payloads, `next_attempt`, terminal status | Ensure the resumed `SSWResult` has the same observations and records as one-shot execution. Omitting rejected minima or failed records changes the public result. |
| `rng` | `numpy.random.Generator.bit_generator.state` converted to JSON-safe lists/scalars, plus bit-generator name | Prevent resampling. Restore only against an allowlisted NumPy bit-generator implementation; reject unknown names. |
| `config` | All `SSWConfig` scalar fields, including optimizer, frame, width, HVP/rotation tolerances, stage budgets and height budget | Prevent silent parameter drift. Dataclass reconstruction must validate values before any PES request. |
| `ls` | mode (`paper`/`native`), explicit settings, frozen potential arrays/metadata, response-controller fields, and native cycle state if used | Prevent reinitializing pair tables or response steps. Native and paper schemas must remain separate. |
| `accounting` | requests already charged, per-segment request total, initial-quench requests, attempt index/request ranges | Make split cost additive and distinguish prior work from resumed work. |

Atoms, NumPy arrays, tuples and mappings must use explicit JSON primitives
(nested lists, numbers, booleans and strings). `SSWStep` and `QuenchResult`
must be encoded by named fields; their current object fields include nested
atoms, telemetry and diagnostics and cannot be recovered safely by generic
`asdict` alone. Gaussian terms inside a completed record should be represented
as numeric center/direction/width/weight arrays, never as class names.

## LS-specific serialization boundary

Paper LS can encode the frozen potential's immutable identity and arrays:
numbers, cell, PBC, pair indices, reference distances, strengths, `xi` and
energy-filter entries. `LSResponseState` needs its target, learning rate,
completed update count and last response. Native LS additionally needs the
explicit `NativeLSSettings`, frozen MIC pair data and the full
`NativeLSCycleState` fields including table, old bond count, cycle parameters,
saved note table/count/response and current step. These fields are visible in
`standalone/softening.py:227-265` and `standalone/native_ls.py:170-264`.

The resume loader must validate that the supplied settings and the serialized
geometry contract match before any request. It must not call
`initialize_native_ls`, `FrozenBondSoftening.from_atoms` or the initial true
quench during resume: doing so would rederive state, spend requests and break
one-shot equivalence. If a field needed to reconstruct the exact controller
state is absent, continuation must fail explicitly as `unsupported_checkpoint`
rather than silently restart.

## Split versus one-shot equivalence

For a fixed calculator and deterministic optimizer, the target is numerical
trajectory equivalence at every completed attempt: same sampled directions,
Gaussian history within each attempt, landing/MC outcomes, LS table update,
records, minima order and cumulative request counts. This requires more than
serializing `current`: RNG state, attempt numbering, response state, frozen LS
state, and all prior result records are part of the observable.

The equivalence claim is conditional. It is unknown whether an external
calculator is deterministic across process boundaries, whether optimizer
floating-point reductions are identical after reconstruction, and whether a
calculator cache affects request accounting. The first implementation should
therefore report `same_process` and `fresh_process` replay separately. A
successful JSON round trip proves state reconstruction, not scientific
validity or global-search quality.

## Cost accounting

The first segment charges the initial true quench plus completed attempt costs.
Each resumed segment charges only requests made after loading the checkpoint;
the returned cumulative total is `checkpoint.accounting.requests_before +
segment_requests`. The initial-quench cost must remain in the overall campaign
manifest but must not reappear in a resumed segment. Failed or censored work
before a checkpoint remains charged and retained in `history`; no retry is
implicit.

## Bounded validation plan

Validation should use saved, already qualified inputs and the same explicit
calculator/backend configuration, with no parameter tuning:

1. **Cu13/EMT:** run a short ordinary SSW and paper/native LS case in one shot,
   split after one completed outer attempt, serialize/load JSON, and compare
   records, selected current, best, LS update fields and additive request
   counts. Use a same-process replay first, then a fresh-process replay if the
   calculator contract can be reconstructed.
2. **C4H6/GFN2:** repeat the paper-LS and native-derived LS split at the existing
   two-seed development settings. Compare torsion/fragmentation diagnostics and
   full saved record order; do not treat same coordinates as chemical identity
   without the separate graph/physical checks.
3. **Fixed-cell Cu crystal/EMT:** exercise periodic paper LS with the fixed-cell
   translation-only contract and verify cell/PBC identity, periodic frozen-pair
   reconstruction, landing order and request ledger. Native MIC behavior should
   be a separate explicit case because it is not periodic-image parity.

The minimum acceptance criteria are: no second initial-quench request on
resume; identical RNG-derived attempt seeds; identical LS response/table state;
same ordered records/minima in same-process replay; and exact additive cost
reconciliation. Force certificates and chemical/Hessian validation remain
separate checks.

## Recommendation

This is a useful shared capability, but it should be implemented only after the
outer result schema is frozen. It is a moderate serialization/API change rather
than a kernel change, and should not be mixed with new search heuristics. If the
project cannot guarantee an allowlisted calculator reconstruction and stable
same-process replay cheaply, defer it: a misleading continuation claim would
be worse than the current explicit re-quench behavior. The existing fixed-cell
SSW/LS algorithm does not depend on this checkpoint to run or to satisfy its
current scientific evidence boundary.
