# Fixed H4 versus H8 equal-budget gate

## Question

Does limiting each cumulative-Gaussian uphill walk to four micro-steps produce
more useful potential-energy-surface exploration per force evaluation than the
current eight-step horizon?

This is not an early-stopping experiment. `H4` and `H8` are two discrete,
predeclared action fidelities. Each action is propagated to its fixed horizon
or an existing physical/geometry termination, then receives one true-PES
quench.

## Physical hypothesis

U-O1 showed that the present walk usually reaches its recorded energy scale in
the first micro-step, while biased proposal relaxation consumes 61.5--75.1% of
the campaign budget. Later micro-steps may still change the basin of attraction.
The unresolved trade-off is therefore

\[
\text{more continuation capacity per action}
\quad\text{versus}\quad
\text{more independently quenched actions per total budget}.
\]

H4 is useful only if the additional completed actions compensate for losing the
last four cumulative-bias propagation steps.

## Frozen comparison

- Systems: C60, fixed-bottom PdO, and CuO.
- Seed: 49.
- Arms: `H4` and `H8`.
- Total campaign budget: 20,000 force evaluations per system and arm,
  bootstrap included.
- Bootstrap: exactly one true-PES quench per system/seed, reused by both arms;
  its state hash, energy, and FE count must match.
- Starter: `metropolis_chain` with its independent selection RNG.
- Target: current `archive_scaled` controller.
- Frozen: candidate sources and count, direction HVPs and static score,
  cumulative Gaussian form, trust feedback, local softening, proposal optimizer,
  true quench, matcher, archive, random seed, calculator, model, precision, and
  all other configuration fields.
- Only permitted effective-config difference besides output paths and remaining
  search budget: `max_steps_per_walk`.

No production default changes in this gate.

## Measurements

For every arm retain the existing typed action history and exact global purpose
ledger. Report:

1. best-energy improvement integrated over the complete 20,000-FE axis;
2. final energy drop;
3. completed actions and new minima per 1,000 FE;
4. global-best improvements per 1,000 FE;
5. duplicate and invalid/rejected action rates;
6. direction, biased proposal relaxation, true-PES check, and landing-quench FE;
7. wall time as an engineering metric, separate from FE efficiency;
8. walk-length and termination distributions.

`observed_max_height_eV` remains a micro-step endpoint diagnostic, not a saddle
height or physical barrier certificate.

## Decision rule

For each system compute `H4 - H8` for gain AUC. H4 advances to a repeat gate
only if:

- gain AUC is strictly higher in at least two of three systems;
- the median three-system gain-AUC difference is positive;
- strict landing validity/certification does not regress materially;
- all purpose ledgers close with `unattributed=0` and each residual is smaller
  than one atomic HVP batch.

Otherwise retain H8 and stop the short-horizon branch. A sign flip does not
authorize H5/H6, adaptive first-passage stopping, target scans, posterior
allocation, or new propagators.

## Evidence ceiling

One float32-GPU seed across three systems is a bounded survivor gate. It can
reject promotion or justify a repeated-seed gate; it cannot prove statistical
significance, model transfer, thermodynamic sampling, or universal superiority.
