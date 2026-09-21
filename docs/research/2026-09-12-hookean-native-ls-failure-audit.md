# Cu111 Hookean/native-LS prequench failure: MIC audit

**Scope:** offline inspection of the saved arm
`research/ga_ssw/evidence/hookean-multicase-20260912/cu111-ls_native/`.
No PES was run for this audit and no default was changed.

## Direct evidence

The saved `result.json` has status `ls_prequench_failed`, 152 paid requests,
one initial converged minimum, and one failed LS attempt. The initial Hookean
objective is finite and certified (`active_fmax=0.0146383`), with the plane
constraint contributing `0.00798808 eV`; the failure is therefore after a
successful initial constrained quench. The saved summary also contains a
runner bookkeeping exception (`ConstrainedSSWResult` has no
`evaluation_requests`); this is separate from the result status and must not be
used as the physical failure cause.

This arm uses `NativeLSSettings(bond_geometry='native-mic')`. In
`pamssw/standalone/native_ls.py`, `_bonds()` selects pairs using ASE
`find_mic` (lines 83–110), while `FrozenBondSoftening.evaluate()` in
`pamssw/standalone/softening.py` calls `find_mic` again on every evaluation
(lines 189–210). Thus the selected pair list is frozen, but the periodic image
used for each pair force is not frozen. The same file explicitly documents that
MIC branch boundaries are not smooth (lines 106–108).

The ledger gives a concrete boundary encounter. For the selected pair `(8,11)`
and the fixed skew cell, the two competing image lengths are equal to about
`2.546513518 Å` at request 8. At request 9, the nearest image changes from shift
`(0,-1,0)` to `(-1,0,0)`; the two lengths are then `2.547931012` and
`2.545111989 Å`. The shift changes repeatedly during the early trial sequence.
The recorded total-force change from request 8 to 9 is `0.010535 eV/Å`; request
12 to 13 has `0.056072 eV/Å`. These are ledger observations, not isolated
finite-difference estimates of the LS force. The later `0.1565 eV/Å` plateau is
the saved bare physical-force diagnostic, not the optimizer's physical-plus-Hookean-plus-LS gradient; the optimizer telemetry ends at
`steps=48` with `gradient_norm=0.0657916`.

There is an explicit alternative in the implementation. With
`bond_geometry='periodic-images'`, `native_ls.py` freezes image records from
`neighbor_list` and constructs `FrozenPeriodicBondSoftening`; its evaluator
uses the stored image shifts. Existing tests cover finite-difference force
consistency and smooth crossing for this alternative in
`tests/standalone/test_native_ls.py` (`test_periodic_image_force_fd_and_frozen_crossing_are_smooth`). The native-MIC failure test in
`tests/standalone/test_native_ls_driver.py` intentionally preserves a
prequench failure and states that native-MIC is distinct from this periodic
image potential.

## Interpretation and boundary

The data establish that this native-MIC arm repeatedly encounters an image
branch boundary while optimizing the LS objective, and that the implementation
can therefore supply a discontinuous force direction at such a crossing. This
is a credible common mechanism that can make an LBFGS prequench fail or stall.
The ledger force is the bare physical force, not the gradient of the optimized
physical-plus-Hookean-plus-LS objective; its magnitude cannot diagnose LS
convergence. The saved optimizer ends at step 48 with objective gradient norm
`0.0657916 eV/Å` and `line_search_failed`.

The saved ledger does **not** establish that the final failure was caused solely
by MIC switching. The trajectory also contains optimizer trials, Hookean forces,
and a later large force/energy change; the bare-force plateau must not be
interpreted as an LS residual.  and no controlled same-trajectory
comparison was run. Switching to `periodic-images` would change the LS
mathematical potential and is not a native-parity repair. It is already an
explicit research option, so no automatic fallback or default change is
justified by this arm.

The remaining precise evidence gap is a paired offline force-continuity check
at the observed request-8 boundary with the same frozen pair data, plus a
separate native-MIC versus periodic-images run under a predeclared protocol.
Those are future diagnostic experiments, not a conclusion that the native
implementation is generally invalid.

## Follow-up: exact final failed line search reconstructed offline

The newer audit uses the reference arm in
`constrained-gaussian-reference-20260912/cu111-ls_native/` (its full physical
ledger was previously verified identical to the parent). It reconstructs the
last 10 accepted secants, Safe-total direction, each of the 20 halved trial
coordinates, and physical+Hookean+frozen-LS energy from the saved physical
E/F ledger. No new calculator request is used.

- Last accepted request132, accepted step48, soft objective4.309605485316359eV.
- `g dot p = -0.0003122047899513089 eV`, so this is not the non-descent-direction
  branch. Maximum per-atom direction norm0.1424534Å is below the0.2Å step cap.
- Requests133–152 test alpha1 through2^-19. All20 fail Armijo and all actually
  increase the soft objective: first +0.18324253eV, last +5.72974e-10eV.
- Reconstructed coordinates match saved trials within9e-16Å.
- Last trial displacement norm4.96253e-7Å; bare EMT-force difference norm
  3.64249e-6eV/Å, but frozen-LS force difference norm0.0930152eV/Å.
- Pair(8,11) changes nearest image from(0,-1,0) to(-1,0,0) even on that last
  small trial. This is direct evidence of nonsmooth LS image selection at the
  failure location, substantially stronger than the earlier early-trajectory
  observation. It explains why the selected branch gradient predicts descent
  while the tested displacements increase energy. It does not establish that
  no other direction could descend, or that periodic-images guarantees success.

The event is not a failed electronic SCF (the physical model is EMT), not
maxiter300 exhaustion, and not a Gaussian failure. The Python outer driver
requires converged prequench; after line-search failure it stops before rotation.
That continuation rule is an independent implementation choice. Native evidence
in `native-ls-cycle-state.md` includes force OR optsoftmax exits, and does not
justify claiming native LASP demands this same0.03 maximum-atom-force threshold.

Artifact: `final-line-search-audit.json` within the newer reference arm.
Reconstruction script: `research/ga_ssw/audit_native_ls_final_linesearch.py`.
