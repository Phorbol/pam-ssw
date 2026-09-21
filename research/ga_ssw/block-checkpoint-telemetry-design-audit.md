# Block baseline checkpoint and telemetry design audit

This is a zero-PES design audit of the two Fe7C3 block-baseline records. It
does not authorize a rerun, budget extension, or release of a budget-exhausted
state.

## What the current checkpoint preserves

`pamssw/standalone/atomic_climb.py:16-32` defines an explicit
`AtomicClimbCheckpoint`. It preserves the completed-boundary atoms, the fixed
initial direction anchor, contiguous completed Gaussian events, reference
energy, exact config, cumulative request count, and a `pending` diagnostic.
Each completed event contains center, direction, width, weight, biased energy,
force certificate, rotation residual/request count, quench request count, and
`true_energy` (`atomic_climb.py:123-131`). Thus a completed Gaussian boundary
has enough geometry and Gaussian terms to be replayed by
`resume_atomic_climb` (`atomic_climb.py:68-96`).

The pending record is weaker. It is updated through rotation, height, and
biased-quench setup (`atomic_climb.py:103-123`), but the failed quench's final
finite optimizer state is not copied into it. If the oracle raises during
`quench`, `work` is not advanced by the assignment at line 126, the exception
path only appends an error (`134-140`), and the Safe-total internal trace is
lost at the public boundary. The block caller stores the result and last work
(`block_ssw.py:130-152`), but cannot recover the last accepted partial
optimizer coordinate from that result.

For the observed block runs, outer index 0 is cell-only
(`atomic_scheduled=False`) and has five completed cell cycles before a
certified high-energy landing. Outer index 1 is cell plus atomic
(`atomic_scheduled=True`); its five cell cycles complete, then the atomic
climb is cut by the request cap. The atomic result contains 9 completed
Gaussian events for seed 7 and 8 for seed 101; each completed event has a
finite `true_energy`, while the next event remains pending. The atomic result reports
`evaluation_failed` with the budget error, so the available completed
Gaussian boundary can be distinguished from the unfinished Gaussian. Budget
exhaustion must remain a failure/censoring state; it is not a convergence or
release certificate.

## Smallest future observability change

The fixed-cell public `QuenchResult` in `pamssw/standalone/surface.py:76-85`
currently retains only atoms, energy, max force, convergence, optimizer steps,
request count, and surface kind. In the Safe-total branch, the richer
`RelaxResult` is created at `surface.py:137-140`, but only `n_iter` is copied;
the `RelaxTelemetry` (including `termination_reason`) is discarded. The
minimum non-policy API addition is a nullable `telemetry` field carrying the
existing `RelaxTelemetry` object, with no new stopping rule or threshold.

At minimum, serialized consumers need `backend`, `termination_reason`,
`optimizer_success`, `gradient_measure`, `evaluator_calls`,
`backend_evaluations`, `accepted_steps`, `rejected_steps`, and line-search
counts. Keeping the complete telemetry object avoids another lossy adapter.
`pamssw/relax.py:141-176` already constructs these fields and
`relax.py:817-865` supplies the Safe-total termination reason. The
`cell_relax.py:21-23,27-35` path already returns its optimizer result inside
`CellQuenchResult`, so this information loss is specific to fixed-cell
`surface.quench` and its callers.

The direct impact points are the biased atomic quench at
`atomic_climb.py:123-127`, fixed-cell paper quench calls at
`paper_reference.py:206,378,421`, and the fixed-cell/modified-surface users
of `surface.quench` listed by repository search. `block_ssw.py:130-134` should
serialize the telemetry alongside `partial_status` when it serializes an
atomic result. This exposes why a quench stopped and supports deterministic
checkpoint accounting; it does not turn an incomplete partial state into a
landing.

## Design boundary

A future same-budget policy can use a completed checkpoint boundary as an
explicit candidate input for a separately requested true quench, provided the
request ledger has enough budget and the resulting physical E/F/stress
certificate passes. The pending unfinished Gaussian may be replayed from its
boundary, but its last partial optimizer point must not be promoted without a
new explicit checkpoint contract. No heuristic threshold, implicit fallback,
or budget-exhaustion release is required by this audit.
