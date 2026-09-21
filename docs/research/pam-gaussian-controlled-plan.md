# Controlled PAM Gaussian comparison before more stopping-policy changes

2026-09-11. User correction: earlier PAM evidence reportedly favored adaptive
bias over fixed-height and early-exit variants. Audit that evidence before
attributing recent stage-cap gains to a generally better stopping policy.

Definitions matter. The present atomic reference uses fixed width and
force-dependent height, not constant height. A per-Gaussian accepted-step cap,
whole-walk lower-energy release, optimizer energy tolerance, and outer search
budget are four different stopping rules. They must not share an 'early stop'
label in conclusions.

## First experiment: bias construction only

All arms use the same fixed-cell atomic direction solver inside the same
sequential VC block, inputs, model, seeds, final force/stress criteria, history,
cell schedule and total EFS cap. Keep inner fmax=.001 and maxiter300 with strict
biased force convergence. No stage25 option in this matrix. Keep whole-walk
lower-true-energy release and14 Gaussian cap unchanged.

A: existing fixed width .6A + BP-CBD forward-force height.
B: width .6A + PAM curvature-based height.
C: PAM curvature-adaptive width + same curvature-based height.

B-A isolates height rule; C-B isolates width rule (including its dependent
height). These are same-kernel policy ablations, not a full PAM-versus-LASP
benchmark. The production-reference fb02469 and current config default to
per_atom_rms step length; C explicitly selects the existing curvature_adaptive
formula, not the complete production-default walker. No trust feedback,
archive acquisition, new direction selection or extra geometry guards.

PAM core parameters copied from config/walker: target_uphill_energy .6eV;
target_negative_curvature .05eV/A^2; width bounds .15/1.5A; weight bounds0/10eV;
curvature floor1e-4eV/A^2. These are inherited choices for a diagnostic pilot,
not universal physical constants or an optimal Fe7C3 fit. The ideal un-clipped
height sigma^2*(k_inner+k_target) cancels directional curvature at the center;
clipping can prevent the requested curvature. Width uses |k_true| and a
quadratic energy scale, which does not guarantee a barrier crossing or a
bounded actual energy rise. Record raw values and every active clamp.

Remove the known analytic rotation-only curvature from the direction result;
add analytic old-Gaussian directional Hessians to obtain the inner curvature.
The true and inner curvatures must not be conflated. Zero height is permitted
by PAM's source and should not be replaced by a fabricated positive floor.
Use the same unwrapped chart; do not introduce PAM's MIC convention into this
ablation. Finite-difference curvature accuracy remains that of the fixed solver.

Prospective pilot: Fe7C3-80, MACE-OMAT-0-small double, seeds7/101, two outer
steps per arm,2000EFS per arm including independent checks; six arms<=12000EFS.
No outcome-driven retries, cap changes or parameter tuning. Preserve baseline
censoring and all failed attempts. Require exact ledger accounting, fresh true
forces/stress and structural comparison. Report low-energy discovery and
valid distinct landing coverage separately from proposal completion.

## Subsequent stopping tests

After this isolated bias comparison, use a prospective crossed design of bias
policy and inner stopping, retaining all arms rather than choosing a seed-wise
winner. Separate inner bias fmax/maxiter from initial/final true fmax/maxiter
in the API before varying them. Numerical failure is not budget completion.
True certificate thresholds remain common for candidate qualification, even
if a quench search budget is varied. No hyperparameter sweep combining bias,
inner tolerance, final tolerance, directions and cell displacement.

Old PAM full-walker results can provide context but cannot isolate the Gaussian
component if direction, controllers, budget or tolerances also changed. Broader
independent real cases follow a mechanism result; reused Fe seeds are development
evidence only. Do not promote early stopping or this curvature variant now.


## Historical evidence recovered from Git (not current runtime files)

-940cdc5:runs/20260730-uphill-propagation-u0-u1/final_report.md
  (executionff379f3):16 prefix attempts,9 completed matrices,27 arms,4474FE.
  Fixed calibrated/no-feedback neither promoted. C60 aggregate hides late
  rescue and earlier cost/force-certificate regression. The table's certificate
  column0.000 is DELTA against baseline, not zero absolute certification.
-3110f084:runs/20260801-fixed-step-target-gate/conclusion.md:
  fixed0.8eV macro target won5/9 paired blocks, below predeclared6/9 threshold;
 359998FE. A macro target is not a constant Gaussian weight.
-e3f014f:runs/20260801-true-energy-descent-early-stop-gate/conclusion.md:
 24 frozen paths,82 accepted endpoints,4/4 triggered low-energy landings but
 6/26 escape-horizon coverage. One crossing would miss a deeper later basin.
 This is offline counterfactual evidence, not online speedup.

Verbatim historical reports are copied to pam-bias-historical-evidence with
source refs. They motivate retaining PAM as a comparator, not declaring a
universal winner. U0/U1 used proposal fmax.05/maxiter80, true-quench maxiter400
(and material-specific final fmax), unlike current Fe's.001/300. The first new
matrix intentionally holds these numerical conditions constant across arms.

| Control | Current experiment | Changed here? |
| --- | --- | --- |
| Gaussian rotation | same dimer and request/residual settings | no |
| Biased optimization | Safe-total, fmax.001, maxiter300, strict | no |
| Cell-interleave atom relax |25 accepted steps | no |
| True initial/final quench | maxiter300, fmax.001, stress.0001 | no |
| Gaussian count per combined walk |14 | no |
| Whole-walk lower-energy release |same original reference | no |
| Outer trial count and total budget |2 trials,2000EFS/arm | no |
| Gaussian height/width construction |A/B/C | yes |

The first six-arm job1270644 completed (oneV100,371s,11994EFS): all six
combined attempts remained censored at the common search cap. They completed
9/8,11/8,13/12 Gaussians for A/B/C, respectively; this is not a search gain.
Height-only seed101 hit10eV in8/8 recorded stages; C hit neither clamp and
used widths0.275--0.306A and0.182--0.192A. No zero-height cases occurred in the
returned records, so the differing zero-height admissibility rule did not
explain these recorded stages. Pending incomplete stages remain separate.

A prospective matched follow-up is frozen in
research/ga_ssw/fe7c3-pam-gaussian-inner002, job1270704: same three policies and
seeds, only bias_fmax=.02 (existing PAM proposal default), strict maxiter300,
no stage-budget release, finalfmax.001/stress.0001 unchanged. This crosses
policy with inner force tolerance; it is not a sweep of new thresholds.
Source diff confirms unchanged Gaussian, direction and cell kernels, with
only the new inner-tolerance dispatch and telemetry/config fields changed.
The interface and true-quench separation pass36 targeted checks. No second
inner threshold, maxiter or policy tuning will follow based on these results.
