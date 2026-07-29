# Fixed GPU proposal energy traces

This benchmark-local runner replays the already frozen, one-Gaussian proposal
tasks with the shared `maxiter=400` observation cap.  It uses independent
warmed calculators from the committed fixed-replay/G1 helper, retains only
`ase-fire` and `safe-lbfgs-total`. It fails closed on current-run observer
integrity, finite values, exact source/task/model/input provenance, CUDA
availability, and atomic publication. The old cap-400 raw run is a validated
historical comparator, not a deterministic oracle: its call/endpoint
differences are recorded instead of making a current trace invalid.

Each trace point is recorded while the relaxation evaluator already has the
PES components in hand.  `EvalCounter`, trace-record count, and Relaxer
telemetry must close exactly.  Accepted-state labels are postprocessed from
the Relaxer trajectory callback's coordinate hashes, never from a new model
call.  The raw output remains a local artifact and is deliberately not part
of the repository.

Final energy and coordinate reproduction use measured float32 replay-
equivalence limits of `5e-4 eV` total energy and `2e-3 A` maximum MIC-aware
atomic displacement, respectively. These same-stationary-neighborhood limits
only classify cross-process numerical equivalence; they do not alter optimizer
settings, budgets, or any optimizer claim.

## Claim ceiling

This work establishes matching evaluated points and matching callback-observed
accepted states for the frozen replay.  A trace point not marked accepted is
not automatically a line-search rejection, and these labels make no causal
claim about why an optimizer accepted, rejected, or converged a step.
Historical comparator differences likewise do not establish a new optimizer
effect or invalidate a current observer-integrity-closed trace.
