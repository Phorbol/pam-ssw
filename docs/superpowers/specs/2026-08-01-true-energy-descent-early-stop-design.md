# True-PES Descent Early-Stop Design

## Status and scope

This design specifies the research-only **G-E0** gate.  It tests whether an
SSW walk should stop after the first completed outer micro step whose true-PES
energy is lower than the macro-step starter energy.  It does not modify
`pamssw`, production profiles, Gaussian parameters, the direction oracle, the
proposal optimizer, the true-quench optimizer, or the starter selector.

The gate is limited to global-minimum-oriented search.  The proposed rule is
system-agnostic but not thermodynamically unbiased: it deliberately prefers
the first certified energetic improvement and is not a canonical sampling
rule or a reaction-network rule.

## Terminology

One SSW macro proposal contains up to `max_steps_per_walk` outer micro steps.
Each outer micro step:

1. selects or transports a direction;
2. adds one Gaussian bias;
3. applies the explicit displacement;
4. relaxes on the accumulated biased PES;
5. evaluates the resulting state on the true PES.

G-UP1 studied accepted optimizer frames inside item 4.  G-E0 instead studies
the states after complete outer micro steps, between successive Gaussian
depositions.

## Physical hypothesis

Let the true-PES-relaxed macro starter be `x0`, with energy `E0`, and let `xk`
be the state after outer micro step `k`.  Define the descent crossing

\[
E(x_k) < E_0 - \delta_{\rm num},
\]

where `delta_num` is the already configured `dedup_energy_tol`.  No new
threshold is introduced.

If `x0` is a true local minimum and the following true quench is
energy-decreasing, a descent crossing cannot quench back to the higher-energy
starter.  It is therefore a sufficient certificate for a lower-energy basin,
up to numerical and optimizer failures.  It is not necessary: a checkpoint
above `E0` may already lie across a barrier and may quench to a lower basin.

It is also not an optimal-stopping certificate.  Continuing after the first
crossing may find a deeper basin, or may overshoot and lose the first lower
basin.  G-E0 measures both effects without assigning a heuristic weight to
them.

## Existing evidence and motivation

The immutable current-action first-passage corpus contains 24 C60/PdO action
paths and 82 accepted outer-micro-step endpoints.  Fifty-nine preregistered
horizon checkpoints were quenched; 51 have valid checkpoint energies.

Among those 51 checkpoints:

- all 6 checkpoints below their starter energy are certified escapes;
- none of the 20 return-to-starter checkpoints is below the starter;
- only 6 of 26 certified escapes are below the starter before quench.

The signal is therefore precise in the observed sample but incomplete.  It
also exposes the stopping trade-off:

- C60 plateau/D0/seed 43 reaches a landing 7.08 eV below the starter at h1,
  but the h8 landing is 0.18 eV above the starter;
- C60 plateau/K4/seed 44 reaches a landing 7.01 eV below the starter at h2,
  while continuing to h4 reaches a still lower landing, 9.03 eV below the
  starter.

These observations motivate a complete first-crossing audit; they are not
treated as fresh confirmatory outcomes.

## G-E0 data flow

### Immutable inputs

G-E0 consumes the hashed raw evidence and structure files from
`runs/20260731-current-action-first-passage/output-v3/`.  The runner must
validate:

- the tracked compact evidence SHA256 for the raw corpus;
- all case identities and starter hashes;
- all accepted macro-checkpoint paths against a tracked 82-endpoint G-E0
  manifest created before calculator construction;
- the exact 24-path cohort;
- that rejected attempted states are excluded.

### Energy completion

For every accepted endpoint, G-E0 records `E(xk) - E0` in consecutive outer
micro-step order.  Existing valid energy records are reused.  Missing energies
are evaluated once on the same true PES and charged to
`escape_true_pes_check`.  Invalid or failed evaluations are retained as
unlearnable; they are never imputed.

### First crossing and quench

For each path, the first endpoint satisfying the descent inequality is the
only early-stop candidate.  If its true-quench result already exists in the
source corpus, that result is reused by hash.  Otherwise it receives one new
true quench and validation.

The original terminal endpoint is the counterfactual reference.  Its existing
quench is reused when available; otherwise it is quenched once.  No direction
HVP, Gaussian deposition, biased relaxation, starter selection, or macro walk
is replayed.

### Per-path outputs

Each path records:

- whether and when a first descent crossing occurs;
- crossing true-energy decrease;
- crossing landing label, basin identity and energy decrease;
- original terminal step, label, basin identity and energy decrease;
- outer micro steps that a first-crossing rule would remove;
- whether the rule avoids terminal overshoot;
- whether the rule forgoes a deeper terminal landing;
- new FE by physical purpose and wall time;
- any numerical or structural failure.

## Aggregate interpretation

G-E0 reports a vector of outcomes rather than a weighted score:

- trigger rate over complete accepted paths;
- certified-lower-basin precision among learnable triggers;
- coverage relative to all certified escaped checkpoints;
- number and magnitude of avoided overshoots;
- number and magnitude of forgone deeper terminal landings;
- micro-step reduction under first-crossing semantics;
- true-quench cost at the crossing and terminal endpoints;
- exact new FE ledger and wall time.

The offline gate does not promote a setting.  It only determines whether the
signal is coherent enough to justify a fresh online paired experiment.

## Gate to a future online experiment

G-E1 may be designed only if G-E0 closes its ledger and shows all of the
following:

1. learnable descent crossings in at least two distinct action paths;
2. every learnable crossing quenches to a certified basin below its starter;
3. the observed micro-step saving is nonzero;
4. overshoot avoidance and deeper-basin opportunity cost are both reported,
   without combining them using fitted weights.

These conditions do not promote early stopping.  They only admit a fresh,
paired full-action gate on unseen actions.  G-E1 must freeze starter,
initial-direction action, Gaussian parameters and random stream, and compare:

- the current natural/fixed-cap walk;
- first true-energy descent stopping.

Its primary accounting unit is total action FE:

\[
C_{\rm action}=C_{\rm direction}+C_{\rm biased\ relax}+C_{\rm true\ quench}.
\]

Best-energy-versus-FE, completed macro attempts, duplicate rate, landing-basin
support and wall time are reported separately.  No adaptive wait length,
consecutive-crossing rule or learned stopping model is admitted before this
two-arm comparison.

## Implementation boundary

G-E0 adds only a pure protocol module, a research runner, unit tests, compact
evidence and a conclusion under a dated `runs/` directory.  Production code is
unchanged.

If a later G-E1 supports implementation, the production change must reuse the
`true_after` evaluation that the walker already performs after every outer
micro step.  The macro starter energy must be passed from the already evaluated
archive entry; an extra energy call is not allowed.  The only new behavior
would be a distinct termination reason after the completed micro step.

## Failure handling and stopping limits

- Hash or cohort drift stops before calculator construction.
- Nonconsecutive accepted endpoints stop the affected path as unlearnable.
- Geometry, calculator, quench or certificate failure remains explicit.
- Direction-oracle, biased-relaxation and unattributed new FE must remain zero.
- G-E0 stops before 10,000 new FE or 300 seconds of GPU kernel wall time.
- Partial raw outputs remain ignored; only compact closed evidence is tracked.

## Tests

Pure tests cover:

- exact accepted-path reconstruction and attempted-state exclusion;
- strict first crossing with `dedup_energy_tol`;
- no crossing for equality or numerical-scale decreases;
- reuse of existing energy and quench rows at zero new FE;
- first-crossing selection without looking ahead;
- separate overshoot and deeper-terminal classifications;
- exact purpose-ledger closure.

A one-path GPU smoke must prove that missing energy or quench work is charged
only to true-PES check/quench/validation purposes before the full G-E0 run.

## Claim ceiling

G-E0 can establish whether true-energy descent is a coherent, low-coverage
stopping certificate on the recorded C60/PdO actions.  It cannot establish
online FE savings, optimal stopping, transfer to CuO, equivalence to original
SSW, or production superiority.
