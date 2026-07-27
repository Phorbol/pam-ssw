# Fixed GPU proposal energy traces

This benchmark-local runner replays the already frozen, one-Gaussian proposal
tasks with the shared `maxiter=400` observation cap.  It uses independent
warmed calculators from the committed fixed-replay/G1 helper, retains only
`ase-fire` and `safe-lbfgs-total`, and fails closed unless every row reproduces
the old cap-400 ledger's call count, certificate, termination reason, final
biased energy, and final coordinates.

Each trace point is recorded while the relaxation evaluator already has the
PES components in hand.  `EvalCounter`, trace-record count, and Relaxer
telemetry must close exactly.  Accepted-state labels are postprocessed from
the Relaxer trajectory callback's coordinate hashes, never from a new model
call.  The raw output remains a local artifact and is deliberately not part
of the repository.

## Claim ceiling

This work establishes matching evaluated points and matching callback-observed
accepted states for the frozen replay.  A trace point not marked accepted is
not automatically a line-search rejection, and these labels make no causal
claim about why an optimizer accepted, rejected, or converged a step.
