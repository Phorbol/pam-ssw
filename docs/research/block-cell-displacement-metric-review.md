# Block CBD-cell displacement metric: bounded design review

2026-09-11. Design and bounded implementation of an experimental cell-step
metric in `pamssw/standalone/block_ssw.py`. Default remains the paper rule.
Cu/EMT lifecycle checks passed; the paired Fe7C3 evaluation is separate.

## What the current rule fixes

The current block computes, at `block_ssw.py:109-110`,

```text
DeltaL = 0.15 * ||L||_F * N,
||N||_F = 1
```

so it fixes `||DeltaL||_F / ||L||_F = 0.15` for every direction and cell. This
is exactly the printed 2014 rule `DeltaL = 0.15 sqrt(sum_ij L_ij^2)` with the
current-cycle lattice `L_n`; it is a valid paper-reference baseline.

For row-cell coordinates `R = s L`, the dimensionless right deformation map
is

```text
A = L^(-1) DeltaL,       dR = R A.
```

The current rule gives

```text
||A||_F = 0.15 * ||L||_F * ||L^(-1) N||_F,
```

which is direction- and anisotropy-dependent. For `L=a I`, it is
`0.15 sqrt(3)`, rather than `0.15`, for a unit nine-component `N`. Thus equal
printed `.15` values do not mean equal dimensionless deformation. The current
metric measures absolute lattice-entry displacement relative to the total
cell-entry norm; it does not bound the largest principal/shear deformation.

## Minimal independent proposal

Keep the already selected, rotation-projected unit direction `N` unchanged and
change only the scalar used before `CellChart.unpack`.

The bounded implementation adds one explicit scalar policy,
`cell_step_metric='deformation_rms'`. For a relative RMS distance `rho`, use

```text
DeltaL_Frel = sqrt(3) * rho * N / ||L^(-1) N||_F.
```

This enforces `||L^(-1) DeltaL_Frel||_F / sqrt(3) = rho`, preserving the RMS
relative deformation in the current cell's right-relative chart. The
implementation uses `rho=cell_step_fraction`, so it adds no numerical
parameter. For `L=aI`, every unit nine-entry `N` gives the same displacement as
the existing paper baseline: both have scalar `sqrt(3)*rho*a`.

The existing projection removes tangents `L Omega` for antisymmetric `Omega`
(`cbd_cell.py:28-33, 80-82`). In the row-cell convention these have
`A=Omega`; the proposed scalar change does not alter the direction solver or
its rotational subspace. A symmetric-strain-only metric would be a third
coordinate choice and is outside this bounded comparison.

## Minimal interface and fair comparison

The smallest interface is a named scalar policy called after
`cell_direction()` returns and before `CellChart.unpack()`:

```text
step = displacement_scale(L, N, policy, rho)
DeltaL = step * N
```

The existing paper baseline remains `policy="lattice_frobenius"`, where
`step=.15*||L||_F`; `deformation_rms` is an independent proposal, not native
reproduction. No hidden fallback or new guard is needed beyond the existing
positive-determinant validation. The implementation records the selected
metric, realized RMS deformation, and the principal stretches (singular values
of `I+A`) in each cycle. The helper expects the unit direction returned by
`cell_direction()` and rejects a zero direction; it does not add a second
internal direction-normalization policy.

For the proposed RMS policy, `||A||F = sqrt(3)*rho`; hence `rho=.15` implies
`||A||2 <= .2598`, and `I+A` is nonsingular by the standard `||A||2<1` bound
for one step. This is a one-step mathematical bound, not a guarantee that
repeated accepted moves preserve volume or avoid cumulative compression. The
existing positive-determinant check remains the final geometric guard for both
policies; no new threshold is introduced.

A bounded real-system test should pair the two policies on the same
non-orthogonal anisotropic starting cell, seed, direction input, pressure,
rotation budget, fixed-cell atom-step budget, calculator and outer cycle count.
Record for each first cell move: `DeltaL`, `A`, `||A||_F`, `||A||_2`, singular
values, `det(L_new)/det(L)`, atom-relax request count, and whether the existing
positive-determinant check accepts it. Then run only the already bounded block
end-to-end path and compare accepted landings, physical force/stress
certificates, structural identity, and total E/F cost. A second isotropic cell
is useful as a control for the analytic `sqrt(3)` relation, but does not
replace the anisotropic case.

The comparison is fair only if the initial generated `N`, configuration,
calculator and seed are paired. After the first cell move, trajectories may
naturally diverge and later random directions need not match. If an isolated
scalar comparison is needed, replay the same frozen `N` and stop after the
first move; do not silently force later states to coincide.
No claim about a better metric follows from a single landing. Retain
`deformation_rms` only if its stated deformation invariant is the intended
physical control and paired anisotropic tests show a reproducible,
scientifically useful difference at matched total cost. If it adds no such
evidence, remove the policy and keep the documented paper-reference baseline.

Source anchors: `block_ssw.py:98-110`, `cbd_cell.py:28-33,63-82`, and
`literature/benchmark-sources/vc2014/vc2014-author.txt:180-200,322-336`.
