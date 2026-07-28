# Direction-conditioned checkpoint shooting audit

## Question

For a fixed starter, random seed, direction-allocation arm, bias potential, and
proposal optimizer, does failure arise because the generated direction never
reaches a productive part of the PES, or because the biased uphill propagator
continues past a productive checkpoint or terminates at a poor checkpoint?

This is a mechanism audit. It does not select a production default.

## Frozen cohort

- System: C60 with the same MACE-OMAT-0-small CUDA calculator.
- Starters: `intermediate_accepted` and `plateau_accepted`.
- Seeds: 42, 43, and 44.
- Direction arms:
  - balanced block Krylov, 2 blocks x depth 3;
  - deep block Krylov, 1 block x depth 6.
- Total direction-conditioned uphill trajectories: 12.
- Maximum post-bias macro checkpoints: 8 per trajectory.
- Gaussian bias, local softening, step controller, `safe-lbfgs-total`
  proposal relaxation, and every other search parameter remain frozen to the
  preceding fixed-starter escape audit.
- No starter selector, UCB/TS logic, posterior update, archive feedback, or
  production-default mutation.

The cohort is intentionally restricted to the state/arm contrast that the
preceding experiment identified: neither arm improved the intermediate state,
whereas deep refinement was consistently productive on the plateau state.

## Data acquisition

The existing relaxation-trajectory interface is sufficient. For each proposal
walk, PAM-SSW already writes one
`trial0001_proposal001_stepNNN_proposal_relax.xyz` file per macro bias step.
The last frame of each file is the optimizer result for that bias step. The
walker subsequently applies its existing per-atom `walk_trust_radius` clip
against the original starter and terminates the walk when clipping occurs.
The audit must apply that same core transformation before treating the frame as
the effective macro checkpoint.

The runner shall:

1. reconstruct the locked starter;
2. run exactly one frozen proposal walk with trajectory output enabled;
3. extract the last frame of every macro-step trajectory file and apply the
   existing walker clip, rejecting a clipped nonterminal step;
4. evaluate each checkpoint once on the true PES;
5. strictly quench every checkpoint independently using ASE-LBFGS at
   `fmax=0.01 eV/A`, with the already validated ASE-FIRE certificate fallback;
6. compare every certified landing against the same starter archive.

No callback or checkpoint feature is added to `pamssw/`.

## Accounting

Generation and shooting costs remain separate:

- trajectory generation ledger:
  direction-oracle, biased-proposal-relax, and true-PES-check evaluations;
- per-checkpoint ledger:
  true-PES checkpoint evaluation, strict landing-quench evaluations, and
  validation evaluation;
- shared bootstrap/model-loading costs are recorded once and excluded from
  per-checkpoint costs.

The experiment records all checkpoints. It does not retrospectively charge the
best checkpoint with only its own quench cost or otherwise hide the diagnostic
cost of quenching the remaining checkpoints.

Every completed block-Krylov selection must still consume exactly 12
central-difference HVPs, or 24 force evaluations.

## Outcome semantics

A checkpoint is productive only when:

- its landing has a force-convergence certificate;
- its landing is a new basin relative to the starter;
- its landing energy is more than 1 meV below the starter.

Each trajectory receives one descriptive classification:

- `productive_final`: the final checkpoint is productive;
- `overshoot`: an earlier checkpoint is productive and the final checkpoint is
  not;
- `productive_earlier_and_final`: both an earlier and the final checkpoint are
  productive;
- `no_productive_checkpoint`: no checkpoint is productive.

Continuous energies, descriptors, basin identities, and force-evaluation
counts are retained; the classifications do not replace raw outcomes.

## Interpretation

- `overshoot` is direct evidence that the current propagation or termination
  loses a productive state already reached by the same direction-conditioned
  walk.
- `no_productive_checkpoint` means the sampled direction-plus-propagation path
  never entered a productive quench basin. It does not, by itself, prove that
  the direction alone is defective.
- A checkpoint whose true-PES energy is lower is not automatically selected as
  an algorithm rule. This audit first tests the physical picture without
  introducing an energy threshold or stopping heuristic.

No selector or stopping rule will be implemented from this 12-trajectory
cohort. Any subsequent rule requires a separate preregistered validation on
held-out starters or seeds.

## Execution model

Checkpoint quenches are mathematically independent and therefore
embarrassingly parallel. The initial CUDA runner executes them sequentially
through one shared MACE calculator to avoid introducing process scheduling,
GPU-memory, or batch-relaxer behavior into the mechanism comparison. The
evidence schema preserves checkpoint independence so a later scheduler can
parallelize the same tasks without changing the experiment semantics.

## Verification

- Unit tests lock the 12-case matrix, checkpoint ordering, trajectory
  classification, 1 meV improvement semantics, and closed purpose ledgers.
- The runner pins the full Git commit, source-state hashes, model hash, frozen
  configuration, and per-case random seed.
- Evidence generation rejects missing checkpoints, missing certificate or
  ledger fields, duplicate case keys, or direction selections whose HVP ledger
  does not close. An explicitly uncertified landing remains a recorded outcome.
- Production defaults remain unchanged.
