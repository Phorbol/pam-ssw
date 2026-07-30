# U4 Serial-Gaussian Uphill Mechanism Conclusion

## Question

This audit tests three remaining explanations for C60 uphill stagnation while
holding the starter, selected direction history, modified PES, optimizer, and
strict true-quench protocol fixed:

1. proposal relaxation is censored by `maxiter=80`;
2. retaining all historical Gaussian terms is harmful;
3. applying local softening during proposal relaxation is harmful.

The accepted cohort contains six independently replayed C60 proposal tasks.
Every task originally stopped at `maxiter=80` and contained at least two
Gaussian terms, so every counterfactual changes a real mechanism rather than a
no-op parameter.

## Evidence

- Execution commit: `4207dc4fc8039e459db1d4ad332e1b5492a66399`
- GPU: NVIDIA GeForce RTX 3060
- Total force evaluations: 7,219
  - biased proposal relaxation: 4,399
  - direction oracle: 352
  - escape true-PES checks: 73
  - landing true quench: 2,371
  - post-relax validation: 24
  - unattributed: 0
- Strict true-quench certificates: 24/24
- Recorded execution time:
  - frozen-task generation: 45.93 s
  - independent proposal replays: 43.39 s
  - endpoint validation and true quench: 45.91 s

The independently replayed baseline proposal endpoints differed from the
selection-time endpoints by 0.0017-0.0190 Å RMSD and at most 0.0161 eV on the
modified PES. This limits interpretation of very small endpoint differences,
but it did not obscure the basin-level results below.

## U4-0: Proposal Relaxation Capacity

Increasing the identical task from 80 to at most 300 iterations produced a
proposal convergence certificate in 6/6 pairs. All 6/6 strict true quenches
still reached the same basin as the independently replayed 80-step baseline.
The longer arm consumed 417 additional proposal force evaluations in total.

Conclusion: lack of a proposal convergence certificate at 80 iterations is not
the current basin-discovery bottleneck. Keep `proposal_relax_steps=80`; do not
spend the production budget making every biased-PES micro-relaxation converge.

## U4-A: Gaussian History Retention

Keeping only the newest Gaussian saved 128 proposal force evaluations and
changed the strict landing basin in 3/6 pairs:

- one landing was higher by 1.917 eV;
- two landings were lower by 0.201 and 0.528 eV;
- the remaining three pairs returned to the baseline basin.

Conclusion: Gaussian history is a genuine exploration mechanism, not redundant
optimizer baggage. Newest-only is neither a safe simplification nor uniformly
worse. Retain cumulative history as the production default, but treat
newest-only as a future discrete policy arm if a fixed-budget multi-arm search
is tested. Do not introduce a continuous history-decay parameter from this
evidence.

## U4-B: Proposal-Side Local Softening

Removing local softening only from proposal relaxation left 5/6 landings in the
baseline basin and saved only nine proposal force evaluations in total. The
single changed landing was 0.137 eV higher.

Conclusion: proposal-side local softening is not the present cost bottleneck,
and this cohort does not support removing it. Retain the current `both` scope
until a larger paired search-level experiment shows a reproducible benefit for
`oracle_only`.

## Decision

No U4 arm replaces the current production mechanism. The remaining clean
capacity hypothesis is the number of serial uphill microsteps, not the degree
of convergence inside each microstep. Proceed once to the preregistered
`H=8` versus `H=14` fixed-starter gate. If the longer horizon does not improve
strict landing basins under its additional force cost, stop tuning the current
serial uphiller and return to direction/action posterior design.

Raw evidence is in `output-v3/evidence.json`; the three factor-specific paired
views are `output-v3/u4_0_evidence.json`,
`output-v3/u4_a_evidence.json`, and `output-v3/u4_b_evidence.json`.
