# C60 LS four-operator mechanism gate conclusion

## Decision

G-LS0A closes the current exponential LS operator as a direction-selection
mechanism and does not open the four-arm uphill+quench gate.

The observed curvature reduction comes from the prestrained geometry's **true
PES Hessian**, not from the exponential pair operator.  On the same frozen
candidate pool, neither effect changes the candidate order or selected action.
Consequently the conditional faithful `P_LS=Y` scalar-feedback experiment is
not opened: at the tested initial paper strength, there is no action-level
mechanism for a larger strength to calibrate without assuming the desired
answer.

This does not prove that every possible local prestrain or zero-force operator
is useless.  It rejects the present exponential pair construction and prevents
its promotion through a strength increase alone.

## Frozen execution

- execution commit: `2f42f434653324360583e5b7f58883e8b6c07a76`;
- system: non-periodic C60;
- fixed starters: bootstrap, mid, late;
- direction seeds: 42--44;
- blocks: 9;
- native candidates per block: 4;
- operator rows: 36;
- paper initial strength: `0.1083 eV`;
- exponential decay length: `0.2 * reference pair distance`;
- proposal-side LS: absent;
- true-PES FE: 270 total, 30 per block;
- direction HVP FE: 16 per block;
- pre-relax FE: 13 per block;
- unattributed FE: 0.

The A/B operators share one true-PES central-FD stencil at `x0`; C/D share one
at `xR`.  LS Hessian actions are analytic and add zero MACE FE.  The raw compact
evidence SHA256 is
`fa4cbe93e56405ee9b9198596e8e818441b7dea6b6a153eb9c976786c2cd3e62`.

## Four-operator result

For each frozen direction:

```text
A = H_PES(x0)
B = H_PES(x0) + H_LS(x0)
C = T^T H_PES(xR) T
D = T^T [H_PES(xR) + H_LS(xR)] T
```

| Effect | Sign count | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Exponential operator at `x0`, B-A | 33 harder / 3 softer | -0.2615 | **+0.7620** | +1.5081 |
| Exponential operator at `xR`, D-C | 33 harder / 3 softer | -0.2468 | **+0.7366** | +1.4563 |
| True-PES prestrain, C-A | 36 softer / 0 harder | -2.6550 | **-1.7651** | -0.3103 |
| Operator--geometry interaction | 32 negative / 4 positive | -0.0535 | -0.0240 | +0.0147 |

Curvature units are the existing mass-unweighted Cartesian directional units
used by the production oracle.

The exponential pair Hessian decomposition explains the result:

| Component at `x0` | Sign count | Median |
|---|---:|---:|
| radial pair curvature | 36 positive / 0 negative | **+1.2777** |
| transverse geometric curvature | 36 negative / 0 positive | **-0.5332** |

The transverse prestress term does soften, but the positive radial stiffness
is larger for nearly every candidate.  Calling this term a fixed-geometry
softener is therefore physically incorrect for this cohort.

## Action identity

All four operators give the same complete candidate ordering in every block:

| Comparison with A | Same selected candidate | Same full four-candidate order |
|---|---:|---:|
| B, operator only | 9/9 | 9/9 |
| C, prestrain only | 9/9 | 9/9 |
| D, prestrain + operator | 9/9 | 9/9 |

After rigid-frame transport, selected A/C direction absolute cosines are
`0.999999837--0.999999999`.  Thus the earlier paper-ordered result that reported
9/9 different direction hashes reflected tiny geometry-dependent vector
changes, not a different candidate choice or a materially rotated mode.

The smallest winning score margin is about `0.00967`, while A/B/C/D score
changes are too small to reorder even the full candidate list.  A full
uphill+quench experiment would therefore execute effectively the same action
while paying pre-relax cost; it is not opened.

## What the result says about paper-ordered LS

The initial-strength pre-relaxation is numerically consistent and cheap:

- 85--86 C-C pairs;
- 10 accepted iterations and 13 FE in every block;
- aligned Cartesian RMS displacement about `0.01437 A`;
- `P_LS = 0.00269--0.00276 eV/atom`;
- certified softened-objective force norm `0.0407--0.0487 eV/A`.

It consistently moves C60 to a geometry whose true PES is softer along the
same candidate directions.  That is a real **controlled local prestrain**
effect.  But it does not yet create a different discrete action, and the prior
paper-ordered full-action gate had sign-changing terminal landings.  Therefore
prestrain remains a physical observation, not a surviving exploration arm.

## Metric correction

The run records `||B u||^2 / ||u||^2` under the provisional name
`pair_expressivity`.  Because the redundant pair rows are not orthonormal, this
quantity is not bounded by one; it is an unnormalized pair-radial activity.
It is excluded from the promotion decision.  A future abstention certificate
would need normalization by the largest eigenvalue of `B^T B` or an equivalent
graph metric, but that work is not justified after this gate closes the current
mechanism.

## Research consequence

The following are closed for the current route:

- direct long-search `oracle` versus `none` for the current exponential LS;
- paper scalar-feedback strength calibration to `Y=0.02 eV/atom`;
- pair-specific strength, cutoff, active-count, or adaptive-law tuning;
- CB-ZFLS or local-compliance implementation as an automatic follow-on;
- any claim that the current exponential term is a fixed-geometry softener.

The promoted operational setting remains `local_softening_scope=oracle` only
as the safe way to remove proposal-side LS from historical LS-enabled profiles.
It should not be interpreted as evidence that oracle LS improves directions.
For new mechanism studies, `none` is now the scientific direction baseline.

The main research route returns to stable, physically distinct action families
and first-passage/terminal evidence.  Posterior allocation remains blocked
until at least two such families have repeatable action-level effects.

## Claim ceiling

This is a 9-block, 36-candidate C60 mechanism result at one paper initial
strength.  It is sufficient to close the current implementation path and avoid
an uninformative full-action gate.  It is not a universal rejection of local
prestrain, zero-force negative-semidefinite operators, or the published LS-SSW
method at its fully adapted target.
