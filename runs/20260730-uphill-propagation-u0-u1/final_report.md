# U0/U1 frozen-prefix uphill propagation result

## Decision

Do not promote the fixed calibrated Gaussian. Do not yet replace the current
history-feedback controller with the fully feedback-free curvature-matched
Gaussian.

The feedback-free arm is a justified survivor for one final attribution test on
C60 because it repaired a late-prefix failure and improved the paired physical
metrics at nearly the same proposal-relaxation cost. The aggregate improvement
is nevertheless dominated by that one failure, while an earlier bias-3 task
became more expensive and lost its force certificate. The next experiment must
therefore separate step-width feedback from normalized bias-curvature feedback;
it must not tune another continuous parameter.

PdO provides no evidence for changing the updater. Only the two requested
first-bias tasks were capturable, and all three arms saturated to the same
production `(sigma, weight) = (1.299038..., 10)`. Their residual differences are
an execution-noise control, not an algorithm effect.

## Authoritative evidence

- Artifact: `gpu_screen_v4.json`
- Artifact SHA256:
  `bf1137c42c009ffff8d242837bb5c0d2c742b219b4c2b628925a589933473c4a`
- Execution commit: `ff379f3d1d9c9a35bc6f015d2cb3e2272fe87111`
- GPU: NVIDIA GeForce RTX 3060
- MACE model SHA256:
  `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`
- Protocol schema: 3
- Attempted frozen prefixes: 16
- Completed three-arm matrices: 9 (C60 7, PdO 2)
- Replayed arms: 27
- Total accounted force evaluations: 4,474
- Total wall time: 101.469 s
- Observer-only force evaluations: 0
- Unattributed force evaluations: 0

Schema 3 preserves the costs of successful and censored captures. Earlier
schema-1/2 runs are diagnostic only: schema 1 omitted the shared production
bias-weight bound, and schema 2 omitted the cost spent by censored captures.

## Paired C60 result

Differences below are arm minus `current_full`, averaged over seven completed
frozen prefixes.

| Arm | Final true energy (eV) | Direction progress | Orthogonal displacement | Force evaluations | Certificate rate |
|---|---:|---:|---:|---:|---:|
| curvature matched, no feedback | +0.422 | +0.805 | -1.625 | +2.71 | 0.000 |
| fixed calibrated | +0.520 | +1.548 | -0.270 | +9.29 | 0.000 |

A higher final true energy and larger direction progress are desired for this
conditional uphill step; lower orthogonal displacement and lower cost are
desired. These are a metric vector, not a scalar score.

The aggregate hides strong stage dependence:

- At bias 1, the bounded current and feedback-free tasks are effectively the
  same; differences are at the numerical noise level.
- At bias 3, feedback removal increased cost and one task lost the force
  certificate.
- At bias 5, the current controller produced one non-converged endpoint with
  negative direction progress and very large orthogonal displacement. The
  feedback-free arm recovered positive progress, converged, and used fewer
  evaluations.
- At bias 8, one task improved, while another retained negative direction
  progress in both current and feedback-free arms.
- The fixed arm repeatedly used the maximum weight 10. It was more expensive
  and produced a 9.31 orthogonal-displacement norm in one bias-8 task. Its
  apparent greater uphill motion is not a clean or efficient mechanism gain.

Thus the present evidence identifies a failure mode of the coupled feedback
controller, not a universal win for removing all feedback.

## PdO result and censoring

Six requested PdO bias-3/5/8 prefixes terminated before the requested proposal
task. Those failures still spent 1,030 force evaluations and are included in
the total. The two completed bias-1 matrices used identical bounded Gaussian
parameters in all arms. Relative to current, the feedback-free arm changed:

- final true energy by +0.0010 eV;
- direction progress by +0.0027;
- orthogonal displacement by +0.0785;
- proposal evaluations by +2.

These changes are within the built-in identical-task noise control. No PdO
updater conclusion is supported beyond “the first step is already at the
shared weight cap.”

## Reproducibility boundary

Identical saturated tasks did not replay bit-for-bit on CUDA. In the null
controls, endpoint energies differed by millielectronvolts to a few
hundredths of an eV and optimizer call counts differed by a few evaluations.
Separate complete runs also changed which long C60 prefix was capturable.
Therefore:

- only within-run, same-prefix paired differences are interpreted;
- sub-noise differences are treated as zero;
- no claim is made from cross-run absolute endpoints;
- a full-search promotion requires paired multi-seed evidence, not this local
  mechanism screen.

## Next bounded experiment

Run one C60-only 2x2 factorial on the existing physical variables:

1. step width: current feedback value vs feedback-free base value;
2. normalized Gaussian curvature `weight / sigma**2`: current effective value
   vs curvature-matched base value.

Use only pre-registered bias-3/5/8 prefixes, retain the production weight
bounds, and stop after the six attempted prefixes. Promote a single factor to a
full-walk test only if it explains the late-prefix rescue without reproducing
the bias-3 certificate/cost regression. Otherwise retain the current updater
and move to direction quality rather than tuning more uphill parameters.
