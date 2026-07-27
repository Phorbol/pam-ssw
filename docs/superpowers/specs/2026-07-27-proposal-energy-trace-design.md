# Fixed proposal energy-trace design

## Question

For the already-frozen C60 and PdO one-bias proposal tasks, why does
`safe-lbfgs-total` often require fewer exact evaluator calls than FIRE? In
particular, distinguish monotone accepted-step progress from rejected
line-search work and from convergence to a different stationary basin.

## Boundary

This is an observer-only experiment stacked on the fixed-proposal replay work.
It changes no production optimizer, parameter, proposal objective, task,
calculator, direction, starter, quench, archive, posterior, or budget policy.

The experiment reads the existing version-1 frozen task payloads and replays
FIRE and `safe-lbfgs-total` with the same `fmax=0.05 eV/A` and shared
`maxiter=400` observation cap.

## Recording mechanism

A benchmark-local `RecordingProposalPotential` subclasses the existing
`ProposalPotential`. Its `evaluate_parts` method:

1. calls `super().evaluate_parts(...)` exactly once;
2. records the already-computed true-PES, Gaussian-bias, softening, and total
   energies;
3. records the active-atom maximum total force and a hash of the evaluated
   coordinates;
4. returns the unchanged `RelaxEvaluation`.

The existing relaxation trajectory callback records hashes of initial and
accepted optimizer states. After relaxation, an evaluation is marked as an
accepted-state evaluation only when its coordinate hash occurs in that
callback. No state is re-evaluated for reporting or classification.

FIRE continues through `proposal.evaluate`; safe L-BFGS continues through
`proposal.evaluate_parts`. Both therefore use their unchanged production
execution paths.

## Integrity gates

For every task/backend pair:

- the exact `EvalCounter` total must equal the number of recorded evaluations;
- certificate, final energy, evaluator-call count, termination reason, and
  final coordinates must reproduce the existing shared-cap-400 result;
- source summary and task-spec hashes must match the committed compact
  provenance;
- all recorded energies and forces must be finite;
- no observer path may call the calculator directly.

A mismatch invalidates the trace; it is not repaired by changing optimizer
parameters.

## Outputs

The run stores:

- one JSON trace per system containing every evaluation point;
- a compact summary with per-task call counts, accepted-state counts,
  rejected/non-accepted evaluation counts, endpoint comparison, and trace
  integrity status;
- deterministic curve data for total biased energy, true PES energy, Gaussian
  bias energy, and maximum force versus exact evaluator calls.

The trace describes evaluated points, not physical dynamics. FIRE evaluations
that are absent from its trajectory callback are labelled non-accepted rather
than assumed to be rejected line-search trials.

## Claim ceiling

The experiment may identify where evaluator calls are spent and whether
accepted-state energy decreases are monotone. It cannot by itself attribute
causality to L-BFGS memory, Armijo, or the atomic step cap, because those
controls are not separately ablated. Different final stationary points remain
search behavior, not same-basin convergence acceleration.
