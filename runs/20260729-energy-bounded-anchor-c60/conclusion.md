# C60 energy-bounded anchor direction ablation

## Question and claim boundary

This experiment tests one direction-selection mechanism:

> Can the already-paid anchor-seeded Krylov subspace retain more of the
> random-plus-bond event intent without executing a locally over-expensive
> direction?

The control is the lowest Ritz vector from the same anchor-seeded,
depth-12 Krylov solve.  The experimental arm maximizes anchor overlap inside
that same subspace subject to

\[
\frac{1}{2}\sigma^2 n^\mathsf{T}H_{\mathrm{true}}n
\le 0.8\ \mathrm{eV}.
\]

If the requested RMS step is infeasible in the paid subspace, the experimental
arm uses the lowest true-curvature subspace direction and analytically shortens
the executed step to satisfy the same energy bound.  This adds no HVP and no
scientific tuning parameter.

The starter, random seed, Gaussian-bias propagator, safe-L-BFGS proposal
relaxation, strict true-PES quench, and archive rules are frozen.  No starter
selector, UCB/TS rule, posterior update, reaction-network objective, or
production default is changed.

This is a paired six-case C60 mechanism audit.  It is not a statistically
powered default-promotion result and contains no PdO evidence.

## Locked cohort

- Exact C60 accepted minima from trials 100 and 180 of the locked 200-trial
  trajectory: `intermediate_accepted` and `plateau_accepted`.
- Seeds: 42, 43, and 44.
- Arms:
  - `anchor_lanczos`: lowest Ritz vector from one anchor-seeded depth-12
    Krylov block;
  - `energy_bounded_anchor`: maximum-anchor-overlap direction under the local
    quadratic energy bound in the same paid subspace.
- Each completed direction selection uses exactly 12 central-difference HVPs,
  or 24 force evaluations.
- Meaningful outcome: a strictly certified new basin at least 0.001 eV below
  its starter.

## Execution integrity

- Execution commit:
  `3a94b028b0b2e5da364fdf528ad682b6b34bc179`.
- Completed cases: 12/12.
- Fresh strict terminal-quench certificates: 12/12.
- Quench fallbacks: 0/12.
- Independently revalidated escape and landing hashes: 24/24.
- Force ledgers close exactly for every case.
- Bootstrap, starter-quench, and unattributed force evaluations: 0.
- Process exit code: 0.

## Terminal outcomes

| Starter | Seed | Arm | Landing minus starter (eV) | New basin | Force evals | Direction | Proposal relax | True quench |
|---|---:|---|---:|:---:|---:|---:|---:|---:|
| intermediate | 42 | anchor Lanczos | +1.386383 | yes | 440 | 192 | 211 | 25 |
| intermediate | 42 | energy bounded | +0.000092 | no | 446 | 144 | 275 | 18 |
| intermediate | 43 | anchor Lanczos | +0.000061 | no | 448 | 192 | 230 | 14 |
| intermediate | 43 | energy bounded | +0.000122 | no | 214 | 72 | 110 | 25 |
| intermediate | 44 | anchor Lanczos | +0.000000 | no | 427 | 192 | 206 | 17 |
| intermediate | 44 | energy bounded | -0.000092 | no | 511 | 192 | 277 | 30 |
| plateau | 42 | anchor Lanczos | +0.000214 | no | 363 | 96 | 231 | 28 |
| plateau | 42 | energy bounded | +0.474213 | yes | 302 | 72 | 194 | 29 |
| plateau | 43 | anchor Lanczos | -7.078979 | yes | 443 | 192 | 221 | 18 |
| plateau | 43 | energy bounded | -6.385742 | yes | 605 | 192 | 352 | 49 |
| plateau | 44 | anchor Lanczos | +0.000000 | no | 442 | 144 | 257 | 32 |
| plateau | 44 | energy bounded | -7.079010 | yes | 435 | 96 | 274 | 58 |

`New basin` is evaluated against the starter archive inside each isolated
case.  It is not a cross-case unique-basin count.

Aggregate outcomes:

| Arm | New basin | Meaningful lower basin | Median landing delta (eV) | Best landing delta (eV) | Total force evals |
|---|---:|---:|---:|---:|---:|
| anchor Lanczos | 2/6 | 1/6 | +0.000031 | -7.078979 | 2,563 |
| energy bounded | 3/6 | 2/6 | +0.000000 | -7.079010 | 2,513 |

The extra meaningful event is plateau seed 44.  Both arms still fail all three
intermediate cases.  Plateau seed 43 also shows the limit of a six-case
summary: both arms are useful, but the energy-bounded landing is 0.693 eV less
downhill and costs 162 more evaluations in that pair.

## Direction geometry and constraint behavior

| Arm | Selections | HVPs | Direction FE | Median anchor overlap | Median true curvature |
|---|---:|---:|---:|---:|---:|
| anchor Lanczos | 42 | 504 | 1,008 | 0.2274 | 2.0999 |
| energy bounded | 32 | 384 | 768 | 0.5490 | 4.1667 |

For the energy-bounded arm:

- 30/32 requested steps have a feasible direction in the paid subspace;
- 31/32 selections lie on the active energy boundary;
- 2/32 infeasible requested steps are physically shortened;
- maximum executed quadratic energy is
  \(0.8000000000000002\) eV;
- median exact-anchor curvature is 40.8391 eV/A2.

The mechanism therefore does what it was designed to do.  Relative to lowest
Ritz, it retains substantially more anchor information while reducing the
raw anchor's high curvature by about one order of magnitude.  The higher
selected curvature is intentional: this arm is not another softest-mode
solver.

The lower aggregate direction cost is not a cheaper oracle.  Both arms pay the
same 24 force evaluations per selection.  The energy-bounded trajectories
terminate after fewer selections (32 versus 42), which is an outcome of the
coupled direction-propagation dynamics.

## Total cost

| Purpose | Anchor Lanczos | Energy bounded | Combined |
|---|---:|---:|---:|
| biased proposal relaxation | 1,356 | 1,482 | 2,838 |
| direction oracle | 1,008 | 768 | 1,776 |
| escape true-PES check | 59 | 48 | 107 |
| landing true quench | 134 | 209 | 343 |
| post-relax validation | 6 | 6 | 12 |
| total | 2,563 | 2,513 | 5,076 |

Measured sequential generation time is 42.14 s for anchor Lanczos and 38.34 s
for energy bounded.  Strict terminal quenching adds 2.20 s and 3.47 s,
respectively.  These hardware- and warm-up-dependent times are descriptive;
force evaluations are the comparison budget.

## Decision

1. Keep `energy_bounded_anchor` as an opt-in experimental arm.  The mechanism
   passes its mathematical, budget, and execution contracts and repairs the
   failure mode of using either the raw hard anchor or the softest
   anchor-seeded Ritz vector.
2. Do not promote it to the production default.  The evidence is only six C60
   cases, and the comparison control is `anchor_lanczos`, not the stronger
   detached-Ritz control from the preceding audit.
3. Do not add TS/UCB-like direction learning yet.  The intermediate starter
   remains 0/3 for both arms, so the action set still lacks a productive event
   there; a selector cannot learn an absent action.
4. The next minimal ablation is a direct, same-commit comparison against
   detached Ritz at the same 12-HVP budget.  Only if the energy-bounded arm
   survives that control should the fixed protocol be repeated on locked PdO
   states before any posterior direction policy is introduced.

The machine-readable source of truth is `output/evidence.json`.
