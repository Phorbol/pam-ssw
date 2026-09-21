# Joint VC implementation and current evidence

> 2026-09-10 更新：用户提供的 http://lasphub.com/publication/87.pdf 已成功取得并核验为12页正式排版全文（2,477,395 bytes），存于 `literature/benchmark-sources/vc2014/`。以下此前获取失败的记录保留作历史；正文缺口已解决，SI仍未取得。全文揭示2014为CBD-cell/atomic-SSW分块耦合，见 `vc2014-native-crosscheck.md`；当前joint log-strain方法须保留独立扩展标识。

An independent ASE joint atomic/cell walker is now implemented in
`pamssw/standalone/vc_reference.py`. It does not invoke LASP, Java or PAM's
walker. It reuses PAM Safe-total numerical primitives via a flat-coordinate
adapter. This is a mathematically explicit VC extension, not native schedule
parity or production search validation.

## Objective and coordinate contract

At each outer proposal, set a reference chart at the current accepted crystal:
`R=X exp(S), H=H0 exp(S), q=(X,L*s6)`. S is symmetric with an orthonormal six
component basis; L is explicitly supplied in Angstrom. Keep this chart fixed
through rotation, all Gaussian deposition, biased relaxation and true quench.
After all biases are discarded, the next proposal may start its own chart.
This avoids carrying the previous reference's accumulated metric distortion
into a fresh proposal; it does not reduce or rewrite a live Gaussian history.

Random directions, the plane dimer, projected Gaussian gradient and Safe-total
updates all include atomic and strain components. The raw surface is E+pV,
with full Frechet pullback of ASE stress; no double-counted atomic virial.
Global atomic translations are projected; a translation-invariant physical
potential is required. Symmetric strain excludes pure cell rotation, but does
not perform lattice basis reduction. Positive determinant is not a physical
validity certificate.

Width and maximum step are q lengths. Rotation bias has energy/length squared
units. L controls the relative atomic/cell proposal metric and has no claimed
universal optimal value. Existing research width .2 Angstrom, rotation bias100
and other defaults are explicit development settings, not oxide-calibrated
physical constants. Pressure and stress tolerances use eV/Angstrom^3.

## A failed landing exposed a convergence-contract error

The first fixed-reference implementation ran three Cu4 EMT proposals with
4254 E/F/stress requests, two valid landings and one failed true certificate.
That failed landing had raw fmax .0201906 eV/Angstrom despite generalized norm
.00466198 below .005. Stress .00027197 eV/Angstrom^3 was within tolerance.
The unreduced cell condition number was91; it is not by itself evidence of
physical collapse. A separate single-landing stricter quench needed8 requests
and passed fresh physical checks; full diagnostic cost11. See
`vc-cu-failed-landing-diagnosis.md` and preserved original evidence.

True-quench stopping now directly checks physical fmax/fmax_target and
maxabs(stress+pI)/stress_target at the accepted q. The generalized optimizer
accepts an optional coordinate-dependent convergence norm. Its rejected line
trials cannot replace the accepted-state certificate. Per-quench evaluation
records cache only the physical convergence scalar by exact q, with no extra
oracle evaluations. An independent post-quench certificate is still obtained.
Modified-surface optimization retains the declared generalized gradient norm.
This removes coordinate scaling from physical acceptance rather than guessing
a tighter universal generalized tolerance.

## Bounded Cu4 evidence after the correction

Three steps at seed3, L3.6:415 search +4 fresh validation requests; no failed
proposal. This also starts a fresh chart between proposals; it is therefore
NOT a controlled speedup against the first fixed-reference run.

A separate same-seed17 development sensitivity uses L1.8/3.6/7.2, three
proposals each. Results and every cell/direction are preserved under
`research/ga_ssw/evidence/joint-vc-cu4-l*-seed17/`. This probes dependence on L;
it is not evaluation data for selecting and advertising an optimal L.
Physical stationarity, structural novelty and finite-cell stability are
separate checks. No accepted new phase or global-search efficiency is claimed.

The AlOH26 MACE E/F/stress preflight and official TiO2 SI coordinate retrieval
provide material inputs for the next layer. The MACE process requires
PYTHONNOUSERSITE=1 in this installation; otherwise user packages break an
unrelated matplotlib/scienceplots import. No environment packages were changed.

## Material-scale end-to-end results and independent qualifications

Uploaded AlOH26 / MACE-OMAT-small completed one full 14-Gaussian path and true
quench:1748 search requests +2 fresh certificates,391.79 search seconds on one
CPU thread. The higher-enthalpy landing (+0.5915885 eV) was rejected at300 K;
the current structure remained initial while the certified landing was retained.
Both have small physical forces/stresses. Independent periodic neighbor analysis
finds two H atoms change their nearest oxygen partners, with no near-zero overlap
and all H remaining singly O-coordinated across the declared distance range.
Both81-dimensional finite-cell joint Hessians are positive, including lowest-mode
step-size checks, at342 additional MACE requests. This supports a chemically
nontrivial candidate landing; it is not a new-phase or DFT stability claim.
See `aloh-vc-landing-diagnosis.md`. This run retains its pre-physical-stop source
snapshot; it passed physical certificates already and was not rerun to improve
its search result after the subsequent convergence-contract change.

Official-SI rutile TiO2 / MACE using the updated driver completed one joint step
with173 search +2 fresh requests in13.72 seconds. It accepted a0.1219813 eV lower
landing. Initial and landing physical certificates both pass. The independent
39-dimensional joint Hessians have positive lowest eigenvalues0.13157 and1.30071
eV/Angstrom^2;170 additional requests include finite-difference sensitivity.
The high-symmetry-saddle explanation is not supported at this finite-cell
resolution. Structural identification remains separate. See
`rutile-joint-vc-hessian-qualification.md`.

Anatase and TiO2-B SI inputs also have same-PES reference quenches, costing
10+1 and12+1 search/fresh requests respectively; these are reference preparation,
not SSW escape successes. Three raw TiO2 inputs passed stress finite differences
in39 separate MACE evaluations, documented in `omat-tio2-stress-preflight.md`.

Cu strict qualification now covers eight minima from two three-step runs:
all have positive joint Hessians at two finite-difference lengths. Six post-move
landings match a separately relaxed hcp reference in species-resolved periodic
neighbor shells and volume; initial fcc and hcp groups are distinguishable.
The538 qualification +7 hcp-reference calls are separate from search costs.
See `vc-cu4-small-cell-minima.md`; finite-radius matching is not full isomorphism
or proof of experimental Cu phase stability.

Current validation:195 standalone/reproduction tests passed in the combined
run; its one optional machine-code MC test was then enabled with the uploaded
ELF path and passed independently (196 distinct tests passed). Experimental search validity is supported by the separate
physical studies above, not inferred from this test count. No HPC/GPU production
campaign was launched and no stable branch was changed.

Zero-oracle periodic StructureMatcher checks subsequently distinguish the rutile
landing from its initial and the quenched anatase/TiO2-B references at four
explicit tolerance levels, with scale=False. All12 same-structure transformation
and tiny-perturbation controls pass. This strengthens the unassigned-candidate
label but does not name a new phase; see `tio2-vc-structure-identity.md`.

## Next executable comparison and remaining native source gap

A Cu4 three-arm harness now runs fixed-cell, fixed-cell plus posterior joint
quench, and joint-VC proposals with separately charged common initialization,
all failed calls, preserved valid rejected landings and external MC after the
appropriate quench. Its208-call wiring check and two contract tests pass.
The complete pre-execution source/configuration and calibrated FCC/HCP identity
recipe are frozen in `research/ga_ssw/prospective/vc-cu4-three-arm/`.
The proposal is8 new seeds ×3 arms ×10,000 requests, total240,000, one CPU thread,
no GPU/Slurm, with a shared3,600-second cooperative search deadline. It is
prepared, not launched; see `vc-three-arm-runner-contract.md`. The modified
relaxation norms differ between coordinate spaces; this compares the complete
configured methods, not an isolated single-coordinate intervention.

A bounded native follow-up also resolved the update_forcepara diagnostic:
`maxstress=abs(trace(stress)+3*externaltp)*160.2176565/3`. The name therefore
must not be read as the maximum full tensor component. Its actual stopping
consumer is not yet closed, so this does not show that native convergence
ignores shear. Typed DWARF fields improve provenance but do not yet resolve
cell-coordinate conversion or its conjugate force. See
`vc-native-coordinate-followup.md`. Missing2014 VC full text and129KB SI are
still the exact requested primary-source gap for coordinate/schedule parity.
