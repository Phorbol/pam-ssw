# Independent VC / VC-LS / RC-VC consistency review

2026-09-10. Focused source review of the physical and biased objectives, active freedoms, certificates, failure costs and chart lifetime. No frozen material sources/evidence were modified, and no new material search was run. One exact redundant-topology bug was reproduced and fixed; a separate finite-rotation chart limitation remains open.

## Confirmed bug fixed: a counted torsion that moves no atom

Before this review, `RigidChainChart` accepted a nonlinear root `(0,1,2)` and child `(1,2)` sharing joint `(1,2)`. The child's complete subtree then contains only the two rotation-axis endpoints. Rotating it changes no Cartesian position at any angle: its Jacobian column is identically zero. Nevertheless it contributed one coordinate to the dimer/Gaussian driver. Such a coordinate can receive an artificial Gaussian displacement with no physical escape, which violates the intended degree-of-freedom contract.

A regression first failed because the input was accepted. `rc_geometry.py` now rejects a joint whose entire descendant atom set lies in that joint's two endpoint indices. This is a topology identity, not a tuned rank tolerance or a new heuristic. It does not reject an intermediate two-atom body when later descendants contain additional moving atoms. Valid existing articulated inputs retain their map and Jacobian. All20 directly affected chain/forest/RC-VC geometry and driver tests pass.

This fixes the proved structural case. It does not claim that every possible geometric redundancy is now detected: an entire subtree with additional atoms exactly collinear with the axis, repeated dependent joints, and other singular assembled topologies require separate rank-aware domain handling.

## Important open limitation: angle-axis rank loss can cause false reduced convergence

The raw finite map deliberately evaluates full rotation vectors, including full turns, and its finite Jacobian remains mathematically correct. Correct derivatives do not imply a nonsingular optimization chart. For a nonlinear three-atom root, the retained three root-rotation columns have singular values

```
p=0:      [0.5773503, 0.5, 0.2886751]
|p|=2pi: [0.4082483, 2.81e-17, 1.72e-17]
```

in the inspected Lrot=2 Å/radian example. At a full-turn chart singularity, two physical rotation directions disappear to first order. A small generalized gradient can then fail to establish stationarity even on the intended rigid-body manifold. Dimer refinement and biased stopping could be misled before the final quench. The unrestricted final physical-force/stress certificate is a useful independent safeguard, but it does not repair the preceding escape trajectory.

No clipping, diagonal floor or live Gaussian rebase was added. A principled repair needs an explicitly bounded nonsingular root-rotation chart/domain and failure semantics at its boundary, or an exact rechart plus transformation of the entire bias objective. Silently resetting the rotation vector while retaining linear projected Gaussians would change the objective. The raw geometry tests at2pi should remain distinguished from a driver's optimization-domain validation. No archived real trajectory was proved to reach this singularity in this review.

## Gradient and pressure review: no sign inconsistency found in the declared domains

For row-cell `L=L0 exp(S)`, full-atomic VC uses `R=X exp(S)`. `vc_geometry.py` computes atomic gradient `-F exp(S).T` and the Frechet-adjoint cell gradient of `V(sigma+pI)`, with the explicit strain-length scaling. This corresponds to affine atomic motion under a cell variation. The sum of six symmetric strains omits only global cell orientation, not a physical symmetric deformation. Projecting uniform X translations is legitimate only under the explicitly assumed global translation invariance; it is not a projection of physical stress or of internal relative atomic motion.

`FrozenPeriodicCellSoftening.evaluate_stress` differentiates its frozen image-resolved distance vectors. For exponential pair energy with radial derivative `-B/(xi*r0)`, its affine stress contribution is `-sum[B/(xi*r0*r) * d⊗d]/V`. The code has this negative sign and includes self-image stress. Pair identities/reference distances stay fixed during the biased path; the current cell enters the image vectors. Thus adding LS energy, force and stress before the same VC pullback is consistent. Replacing that contribution by atomic forces alone would be wrong; the current code does not do so.

RC-VC has a different atomic cell map: centers are affine but interiors stay rigid. It correctly uses

```
d(E+pV) = -F:(dR-R L^-1 dL) + V(sigma+pI):(L^-1 dL),
```

including the nonaffine force correction. Omitting that term would double-count internal affine work through stress; the current implementation includes it. Pressure contributes through volume only. The existing actual EMT nonzero-strain checks include positive and negative pressure and show that stress-only gradients fail while the full derivative agrees; this review independently followed the derivation rather than inferring correctness from green tests.

No E/F/stress mismatch was found in these three declared maps. Native RC Kabsch/lambda semantics remain different and are not established by this result.

## Biased stopping versus true certificates

- Joint VC and VC-LS minimize the combined scalar enthalpy-plus-bias objective in their fixed chart. The Gaussian gradient is added to the same transformed physical/LS gradient. Proposal `gradient_tol` is a generalized modified-gradient criterion, not a physical force or stress certificate.
- The joint VC final `true_minimize` calls the shared physical `relax_cell_coordinates`; it does not close over softened `bare`. Its cache records full physical per-atom force and residual `sigma+pI`, and final `certify` evaluates the bare physical oracle again. A returned optimization point with an acceptable physical gradient is still rejected if the optimizer reports failure.
- RC-VC calls `cell_quench` for both initialization and final landing, releasing all rigid constraints and allowing all atomic/symmetric-cell DOFs. It requires both numerical success and fresh physical force/stress checks, so a zero reduced torque alone cannot certify a landing.
- A certificate establishes the reported stationarity tolerance, not positive Hessian, basin identity, chemical integrity or model validity. The recently added strict Cu refinement remains a separate diagnostic and is not part of these search checks.

## State, MC and failure review

Joint VC creates a new chart from the selected current minimum at the next outer step, after the previous Gaussian list is discarded. LS prequench changes atomic coordinates only at fixed cell; its physical energy response is therefore also its enthalpy response because pV cancels. The LS table is frozen through the entire subsequent climb, then rebuilt through the response updater at the selected structure after MC. Rejected landings do not become the next reference. The first-rotation anchor is kept through a proposal.

RC-VC likewise freezes its chart and body geometry while bias history exists, uses E+pV for MC, and rebuilds from the selected next landing. It does not wrap individual atoms. Continuous molecular lifts and chemically appropriate persistent body membership remain caller preconditions; a topology-changing true quench can invalidate the physical meaning of the next rigid partition even if the coordinate code still runs. No automatic inference is hidden in this workflow.

All outer request totals use the supplied counted physical oracle, including failed line searches/finite differences and fresh certificates. A local optimizer error does not certify its last trial. LS prequench failure retains its paid requests and terminates the LS lifecycle; LS update failure preserves already obtained candidates. Run-level `completed` in joint VC means its declared attempts ended, not that every attempt was valid; per-step status/certificates are necessary for scientific statistics.

Diagnostic completeness is uneven: the atomic height-policy path now saves its prepared objective before a quench exception, while the joint VC event still records most details after `minimize` returns and some preparation-time exceptions retain only a generic error. Aggregate E/F cost remains charged, but frozen objective reconstruction can be incomplete. This is an evidence/replay limitation rather than a demonstrated wrong physical step; no unrelated logging rewrite was made in this bounded review.

## Changes and validation scope

Only production change: the structural axis-only-subtree rejection in `rc_geometry.py`. Regression: `test_axis_only_subtree_rejected_as_identically_zero_torsion` in `test_rc_geometry.py`. No optimizer, metric, pressure, LS, MC or coordinate-update policy was altered. No frozen campaign source or evidence was touched. Numerical test results supplement the derivation and exact counterexample; they do not replace real-system effectiveness evidence.

Fresh combined VC, VC-LS, chain, forest and RC-VC numerical/driver selection: **39 passed** in2.07 seconds, with existing ASE/NumPy deprecation warnings.

The VC-LS history-capacity forwarding audit is now closed for the public
path. `vc_reference.prepare_ls_step` passes `config.lbfgs_memory` through the
LS preparation and subsequent quench. On the default Cu4/EMT path, the
before/after traces are identical at 39 E/F requests per run (78 combined
requests). The explicit `lbfgs_memory=23` regression first failed on the
unwired path and then passed after the forwarding fix; 16 focused tests now
cover the explicit value and the surrounding VC-LS lifecycle. This is a
parameter-chain regression result, not a search-effectiveness claim or a
change to the default value.
