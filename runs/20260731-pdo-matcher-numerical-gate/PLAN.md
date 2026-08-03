# PdO matcher/numerical ambiguity gate

## Question

Why did five certified PdO checkpoint landings receive
`AMBIGUOUS_MATCH` in the current-action first-passage gate?

The first-passage classifier deliberately required agreement between:

1. the current archive matcher, which combines `dedup_energy_tol` and indexed
   MIC RMSD; and
2. the existing invariant descriptor threshold
   `min_escape_descriptor_delta`.

This gate decomposes those signals without changing any threshold.

## Frozen evidence and cost

- Input: all five `AMBIGUOUS_MATCH` rows from
  `20260731-current-action-first-passage/output-v3/evidence.json`.
- Structures: the exact locked starter and recorded checkpoint landing.
- New force evaluations: zero.
- No MACE features, PCA, SOAP, assignment matcher, re-quench, or threshold
  scan.

For each pair report:

- absolute landing energy difference and existing `dedup_energy_tol`;
- exact current indexed MIC RMSD and existing `dedup_rmsd_tol`;
- existing invariant descriptor distance and threshold;
- continuous maximum indexed MIC displacement as a diagnostic only.

The effective PdO case configuration sets `dedup_rmsd_tol = 0.4 Å`
(`dedup_energy_tol = 0.001 eV`, descriptor threshold `= 0.1`). The protocol
always reads those effective values from the frozen evidence rather than
assuming documentation defaults.

## Frozen decisions

- If all five pairs are geometrically different under the existing MIC RMSD
  matcher while the descriptor says same, classify a descriptor collision and
  allow an offline relabel to `ESCAPED_CERTIFIED`.
- If all five are geometrically the same and split only because energy exceeds
  tolerance, open one strict re-quench numerical gate; do not relabel.
- Mixed behavior remains ambiguous and opens no component.
- This cohort cannot authorize a production matcher or threshold change.

## Preregistered movable-region subgate

The first decomposition produced four global geometry splits and one
`energy_only_archive_split`. For only that residual pair, test whether the
current all-atom RMSD is diluted by the 40 frozen bottom-layer atoms:

- recover the exact fixed mask from the frozen PdO input and its existing
  `np.quantile(z, 0.35)` rule;
- compute indexed MIC RMSD separately over movable and fixed atoms;
- apply the same effective `dedup_rmsd_tol = 0.4 Å` to movable RMSD;
- do not use maximum displacement as a decision threshold.

If every residual energy-only pair exceeds the same 0.4 Å threshold on movable
atoms, classify `descriptor_collision_with_local_dilution` and allow an
offline relabel of this frozen first-passage dataset. If every residual pair
remains within 0.4 Å, open a strict re-quench gate only for those residual
pairs. Any mixed residual behavior remains unresolved. This subgate still
cannot authorize a production matcher change: a movable-region matcher would
need a separately powered cross-system validation.

## Preregistered strict re-quench gate

Run this gate only if the movable-region subgate returns
`residual_energy_only_local_same`.

- Cohort: the one residual starter/landing pair only.
- Objective: true MACE PES, with no bias or local softening.
- Optimizer: retain the original `scipy-lbfgsb`; do not compare optimizers.
- Change only `fmax`: original 0.03 eV/Å to strict 0.01 eV/Å.
- Retain `maxiter = 400`, fixed mask, cell, PBC, and the frozen float32 CUDA
  calculator.
- Independently re-quench both starter and landing, because comparing a strict
  landing against a loose starter would confound the energy difference.
- Account every call as `landing_true_quench` and record GPU-kernel wall time.

After both endpoints satisfy the strict certificate, apply the existing
archive identity rule to the strict endpoints:

- `|ΔE| <= 0.001 eV` and indexed MIC RMSD `<= 0.4 Å`: the earlier split was a
  loose-quench numerical artifact; classify `RETURN_STARTER`.
- Otherwise: two strict stationary endpoints remain distinct under the
  existing matcher; classify `ESCAPED_CERTIFIED` for this frozen offline
  first-passage dataset.
- If either endpoint lacks a strict certificate, remain unresolved.

The existing invariant descriptor is reported but is not a second veto in
this gate: four other pairs already demonstrated descriptor collision on this
PdO cohort. No production matcher, selector, direction, or default changes are
authorized.

## Preregistered certificate-rescue closure

Run this closure only because the strict SciPy landing returned
`optimizer_success=True` while failing the independent raw-force certificate
(`0.01332 > 0.01 eV/Å`). Prior frozen C60/PdO evidence already selected
ASE-LBFGS as the strongest single strict-quench arm and FIRE as the only
fallback covering every ASE-LBFGS certificate failure in that corpus.

- Restart both original frozen endpoints with ASE-LBFGS at `fmax = 0.01`,
  `maxiter = 400`.
- Only when an endpoint fails the raw-force certificate, continue from its
  ASE-LBFGS terminal state with FIRE under the same threshold and iteration
  limit.
- Maximum new budget: 1604 force evaluations (two 401-call primary attempts
  plus two 401-call fallbacks).
- Apply the same strict endpoint identity decision defined above.
- Stop after FIRE even if a certificate is still absent; add no third
  optimizer and tune no optimizer parameter.

This closure resolves the numerical label only. It is not a new optimizer
ablation and cannot modify the production optimizer stack or matcher.
