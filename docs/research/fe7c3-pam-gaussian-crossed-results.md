# PAM Gaussian construction and inner force tolerance: controlled Fe7C3 pilot

2026-09-11. Twelve arms, two requested outer proposals each, same Fe7C3-80
initial structure and MACE-OMAT-0-small float64 potential, seeds7/101. This is
a developmental crossed experiment, not an independent test set or full PAM
walker reproduction. The user asked to separate bias construction from inner
and final optimization conditions before drawing conclusions about early stop.

## Implementation and frozen controls

`PAMCurvatureGaussian` supplies either curvature height at the existing width
or curvature height plus width. It uses the existing PAM core formulas, with
historical Gaussian Hessians included in the inner curvature. The rotation-only
quadratic curvature is algebraically removed from the dimer's finite-secant
estimate. This is not an exact physical Hessian measurement. It does not copy
PAM trust feedback, archive control, direction selection or its current default
per-atom-RMS step scaling. Zero height remains allowed by PAM's source.

`SSWConfig.bias_fmax` now controls only biased Gaussian relaxation. None
retains prior behavior. Initial/final true quench and cell partial relaxation
retain the original force tolerance.36 targeted checks pass, including real
Cu/EMT strict biased stages, checkpoint replay, Cu7 vacancy combined VC,
failure handling, and proof that a looser inner threshold does not loosen the
fresh true-force certificate. Existing defaults were not promoted or replaced.

Three bias arms: A fixed width.6A plus force-dependent BP-CBD height (NOT
constant height); B same width plus PAM curvature height; C PAM curvature
height and width. Fixed inherited PAM core settings: target energy.6eV,
negative curvature.05eV/A^2, width bounds.15/1.5A, weight bounds0/10eV,
curvature floor1e-4eV/A^2. These are inherited numerical/search choices, not
physical constants or parameters fitted here.

First matrix: innerfmax.001. Second matrix: only innerfmax changes to.02,
from PAM's existing proposal default. Both require inner force convergence
with maxiter300; neither enables bias_stage_steps. Finalfmax.001eV/A,
maxiter300,stress.0001eV/A^3 remain fixed. Other common settings:14 Gaussians,
Safe-total/history10, five cell cycles, lattice-frobenius fraction.15,25 partial
atom iterations,p0,T300K,two outer proposals,2000EFS maximum per arm including
fresh checks. Gaussian count, true-quench budget and total search budget are
separate controls. No cap extension, retry or outcome-driven parameter change.

## Results

Counts below refer to a completed combined cell+atomic proposal that passes
independent true force/stress checks. Every arm first completed its cell-only
proposal. All returned noninitial landings were uphill and MC-rejected.

| Bias policy | Inner fmax.001 | Inner fmax.02 | Total EFS at.02, seeds7/101 |
| --- | ---: | ---: | ---: |
| A forward-force height |0/2|0/2|1999/1999|
| B PAM height, width.6 |0/2|1/2|1893/1999|
| C PAM height+width |0/2|1/2|1673/1999|

All six.001 arms used1999 charged requests (1997search+2fresh) and were
censored during atomic climbing. Completed Gaussians for seeds7/101 were
9/8 for A,11/8 for B and13/12 for C. More Gaussians are not an efficiency
success on their own.

At.02, B/seed7 landed at+13.796876eV above the initial structure; C/seed7 at
+14.126047eV. Their fresh maxforce values were0.00086134 and0.00084467eV/A,
with maxstress0.000001444 and0.000004111eV/A^3. Each differs from its initial
and cell-only structures, and they differ from each other, under all three
recorded pymatgen tolerance profiles. This is approximate structure identity
and force/stress stationarity, not Hessian, magnetic or DFT qualification.
B is slightly lower in energy; C costs fewer requests. Neither improves the
best energy, and both are rejected by MC.

C/seed101 completed all14 Gaussians at.02, consuming1452 atomic-climb requests,
but its final true quench exhausted the total cap after161 further charged
requests. The optimizer had accepted155 steps, below maxiter300. Its
true_quench_failed label therefore means budget censoring, not evidence that
300 iterations or the final force/stress tolerances are intrinsically wrong.
Do not treat its last evaluated/accepted structure as a certified landing.
The other.02 failures were atomic-stage budget censoring.

## What the parameter records explain

B/seed101 saturates the inherited10eV height limit at every recorded stage:
8/8 at.001 and9/9 at.02. The estimated center curvature after inserting the
new Gaussian, k_inner-weight/width^2, stays positive:3.36--23.36 and
5.21--28.53eV/A^2. Thus clipping prevents the intended negative-curvature
target in this arm; it is not functioning as an unconstrained adaptive height.
This is a local curvature accounting result, not a barrier-crossing theorem.

C reduces widths to approximately.28--.31A for seed7 and.18--.19A for seed101,
with weights roughly1.2--3.1eV and no recorded width/weight clipping. No
returned stage in any arm had zero height; incomplete pending stages are
excluded from these denominators. Consequently the differing source-policy
zero-height admissibility rule did not explain the recorded-stage comparison.
Width and height must be interpreted together; keeping width large can make
an otherwise curvature-matched height impossible under an inherited cap.

The two-seed data support an interaction between bias policy and inner
optimization accuracy. They do not establish a universally better adaptive
rule, an optimal.02 threshold, or a globally superior early-stopping policy.
The historical reports in pam-bias-historical-evidence likewise did not justify
promoting fixed calibrated Gaussian or first-descent early stopping.

## Cost, evidence and decision

Job1270644: oneV100 on4v100n15,371s,11994EFS,exit0.
Job1270704: oneV100 on4v100n06,346s,11562EFS,exit0.
Total23556 charged EFS including26 fresh checks;10 denied cap requests are
preserved but not charged. Registered Fe cumulative99249 (=75693+23556).
Both exact ledger audits report zero reconciliation errors. Wall times include
orchestration/model startup and should not be read as matched-kernel timings.

Evidence roots: research/ga_ssw/fe7c3-pam-gaussian-controls and
research/ga_ssw/fe7c3-pam-gaussian-inner002. Each retains prospective plans,
inputs, source manifests, frozen implementations, per-request ledgers, fresh
checks, Slurm/allocation records, comparison-summary.json and
Gaussian-policy-summary (lowercase gaussian-policy-summary.json on disk).
The numeric crossed table is research/ga_ssw/fe7c3-pam-crossed-summary.json.

Decision: keep strict and PAM-core alternatives explicit and experimental.
Do not declare early stop the main improvement, promote.02, raise the height
cap, or change final criteria. The immediate numerical distinction is now
available as bias_fmax. The existing bias_stage_steps option deliberately
combines a cap with stage continuation; it must not be mislabeled as an
independent strict maxiter ablation. Maxiter and final-force sensitivity were
not measured here. Next work should compare existing PAM cross-step feedback
against its core at fixed numerical settings, then assess complete low-energy
search with a prospectively adequate total budget and independent real cases.
No further tolerance sweep is justified by these two reused seeds.
