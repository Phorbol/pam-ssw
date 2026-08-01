# Uphill Action Observability Design

## Objective

Determine whether the current SSW action kernel stalls because it fails to
produce a measurable true-PES elevation, or because an elevated escape state
still quenches expensively into an old or unproductive basin.

This stage is observational. It must not change direction generation,
Gaussian accumulation, proposal relaxation, true quench, starter selection,
random-number consumption, termination, or force-evaluation accounting.

## Physical scope and claim boundary

The walker already evaluates the true PES immediately before and after each
outer biased-relaxation micro step. Those endpoint energies are not the maximum
energy visited inside the optimizer line search and are not a transition-state
barrier. The trace therefore calls the derived quantity **observed endpoint
height**, not barrier height:

\[
h_{\max}^{\rm obs}=\max_k\{E_k^{\rm before},E_k^{\rm after}\}-E_0^{\rm before}.
\]

For a macro target \(T\), the dimensionless delivery ratio is

\[
r_T=h_{\max}^{\rm obs}/T.
\]

This tests whether the existing propagation machinery realizes its own
nominal energy scale at the micro-step endpoints. It does not prove that
\(T\) equals a physical barrier.

After the unbiased quench, the trace also records

\[
\Delta E_{\rm quench}=E_{\rm landing}-E_{\rm escape},
\]

the landing-quench force evaluations, optimizer iterations, convergence
certificate, new/duplicate basin outcome, and global-best improvement.

## Alternatives considered

### Extend `DirectionRecord`

This would reuse the existing optional direction archive, but a direction
candidate record is not an action record. It would continue coupling
direction-learning state to propagation and landing telemetry and would omit
actions when the direction archive is disabled.

### Add a configurable JSONL tracer

This is operationally direct but adds new configuration fields, file lifecycle
logic, and serialization behavior before the scientific record is stable.

### Typed in-memory action history — selected

Add three immutable result types:

- `UphillStepRecord`: one completed micro-step endpoint pair and its exact
  purpose-resolved cost;
- `UphillWalkTrace`: the ordered step tuple, nominal target, and walk
  termination reason;
- `ActionRecord`: seed, walk trace, escape energy, landing energy and quench
  outcome for one proposal.

`SearchResult.action_history` exposes every completed proposal action. Research
runners decide whether and how to serialize it. Production configuration and
CLI remain unchanged.

## Data flow

1. `_proposal_pool()` provides an empty one-element trace sink when calling
   `_walk_candidate_from_seed()`.
2. The walk records a step only after the already-existing true-PES
   `true_after` evaluation succeeds. Counts are differences of `EvalCounter`
   snapshots, not new evaluations.
3. At walk termination, `_walk_candidate_from_seed()` deposits one immutable
   `UphillWalkTrace` into the sink and still returns the same `State`.
4. `CandidateProposal` carries that trace to the existing true-quench loop.
5. The loop snapshots landing-quench purpose counts, performs the unchanged
   quench, classifies the basin as before, and appends one `ActionRecord`.
6. `SearchResult` returns the ordered action list. Existing consumers that do
   not inspect it see no behavior change.

## Minimal recorded fields

Each completed micro step records:

- trial, proposal and micro-step indices;
- selected direction kind;
- nominal macro target;
- true energy before and after;
- requested and executed displacement scale;
- base and final Gaussian weight;
- true and inner curvature;
- proposal-relax iterations, outcome class and termination reason;
- direction-oracle, biased-relaxation and true-PES-check force evaluations;
- displacement clipping and whether the walk continued or terminated.

Each action records:

- starter archive ID and energy;
- walk termination and ordered steps;
- escape endpoint true energy when observable;
- landing energy, gradient norm, iterations and convergence;
- landing true-quench force evaluations;
- accepted-new-basin, duplicate, global-improvement and rejection status.

No atomic coordinates or direction arrays are duplicated in the action
history.

## Failure handling

- A walk that terminates before completing a micro step has an empty step tuple
  and its explicit termination reason.
- A budget exhaustion during landing quench produces an incomplete action
  record with the consumed quench count and no landing energy, then preserves
  the existing search termination.
- Fragment and energy-sanity rejections remain observable outcomes rather than
  disappearing from the dataset.
- Non-finite values remain rejected by the existing algorithm; result
  dataclasses validate all persisted numeric values.

## Verification

### Unit and integration invariance

- A deterministic analytic walk must return exactly the previous state and
  purpose ledger while adding the expected trace.
- Running the same analytic search twice with trace inspection versus ignored
  trace must produce identical best state, best energy, archive, action count,
  RNG-dependent choices, and evaluation counts.
- Empty-step termination, duplicate landing, new basin, rejected landing and
  budget exhaustion are covered explicitly.

### U-O1 observational GPU gate

Run the frozen production profile for C60, PdO and CuO at seed 49, Metropolis
starter, archive-scaled target and 20,000 force evaluations per system. This is
one arm per system, 60,000 FE total. It does not compare a new algorithm.

Report continuous distributions and the exact target-attainment split
`r_T >= 1` versus `r_T < 1`; the threshold 1 comes from the algorithm's own
nominal target and is not fitted. For each group report new-basin rate,
global-improvement rate, duplicate rate, landing-quench FE and landing energy
drop. Do not promote a new controller from one seed.

The stage closes with one of three evidence-bounded diagnoses:

1. nominal target is usually not delivered at observed endpoints;
2. target is delivered but elevated endpoints mostly quench to duplicates or
   require disproportionate landing work;
3. delivery and landing behavior are mixed, so a scalar target alone does not
   explain the plateau.

Only diagnosis 1 can admit a later propagation-control gate. Diagnosis 2
returns priority to direction/basin targeting. Diagnosis 3 requires a larger
action-labelled dataset before posterior or quadratic-policy work.
