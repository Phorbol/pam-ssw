# DFT variable-cell relaxation: QE, VASP and ABACUS

**Date:** 2026-09-12
**Scope:** documentation and public-source comparison for a portable ASE VC-SSW/Safe-total design. No production code, PES run, GPU job or HPC job was changed or started.

## Conclusion

The difficult part of cell relaxation is not one universal “bad condition number”. A variable-cell problem combines (i) a coordinate and metric choice for atoms versus strain, (ii) physical coupling between internal coordinates and homogeneous strain, (iii) soft elastic or molecular modes, (iv) redundant translations/rotations or symmetry restrictions, and (v) errors in the energy/force/stress oracle. Plane-wave basis incompleteness adds a particularly important, systematic stress error (Pulay stress). A well-conditioned optimizer cannot repair an inconsistent stress sign, an incomplete SCF, or a cell chart with hidden rotations.

The portable minimum is therefore: one declared enthalpy (H=E+pV), one work-conjugate stress convention, an explicit finite strain chart and atomic/cell metric, joint and alternating schemes as separate controls, a safeguarded line search, and a fresh physical force-plus-full-stress certificate. The present standalone implementation already has the right mathematical boundary: `SymmetricLogStrainChart` uses (H=H_0\exp S, R=X\exp S), `strain_length` is explicit, and `vc_reference.py` uses Safe-total with Armijo acceptance. The next test should isolate metric/coupling hypotheses on the same ASE oracle; it should not add optimizer heuristics or claim parity with any native code.

## What the three programs document

| Question | QE `pw.x` 7.5 documentation and public source | VASP public documentation (6.4-era tags; exact patch behavior is not source-verifiable here) | ABACUS public develop documentation (current page; default CG variant since v3.8) |
|---|---|---|---|
| Objective and pressure | `press` is target pressure in kbar; `press_conv_thr` is pressure threshold in kbar. `vc-relax` applies ionic and cell convergence thresholds. The documented thermodynamic interpretation is the variable-cell enthalpy with external pressure; exact internal scaling is implementation-specific. | `PSTRESS` adds external pressure to the stress used by relaxation. The stress is σij = −∂E/∂ηji; positive diagonal means compression and a tendency to expand. Thus the sign differs from ASE's tensile-positive stress and must be converted before forming (V(σ+pI)). | `cell-relax` exposes force and stress thresholds in the input keywords. Public optimization documentation describes ionic/cell optimization, but does not define a single portable tensor formula or a universal coordinate metric. Verify units in the versioned input page before coupling to an ASE calculator. |
| Cell DOF | `cell_dofree='all'` moves all axes/angles; `ibrav` preserves the initial Bravais family; combinations select/fix components. | `ISIF` separates positions, shape and volume: 2 positions only; 3 all; 4 shape at fixed volume; 5 shape only with ions fixed; 6 shape+volume with ions fixed; 7 volume only; 8 ions+volume at fixed shape. `Selective dynamics` further masks ionic DOF. | `fixed_axes`, `fixed_ibrav` and `fixed_atoms` provide cell/atom restrictions. The public page describes cell relaxation but does not establish that all six independent symmetric strains are always active under every constraint. |
| Atomic/cell coupling | For `cell_dynamics='bfgs'`, QE requires/assumes `ion_dynamics='bfgs'`; both are handled as a variable-cell BFGS route. Damped Parrinello-Rahman and Wentzcovitch routes are separate. | `IBRION` selects the ionic optimizer while `ISIF` selects cell DOF. The CG documentation says the initial direction includes forces and stress, then performs trial/corrector line search with energy, force and stress. This is coupled in the active generalized DOF, but VASP does not publish the complete internal cell coordinate map. | `relax_method='cg 2'` jointly optimizes ions and cell with line search. `cg 1` is explicitly nested: fixed-cell ionic relaxation followed by a cell update. `bfgs`, `lbfgs`, `sd`, and `cg_bfgs` are available; the page calls `bfgs 2` the recommended inverse-Hessian update. |
| Optimizer principles | Cell BFGS is quasi-Newton. `damp-pr` is damped Beeman dynamics of Parrinello-Rahman extended Lagrangian; `damp-w` is damped Beeman dynamics of Wentzcovitch extended Lagrangian. Public input does not promise a common line-search implementation for these modes. | `IBRION=2` CG uses a trial step, recomputes E/F/stress, fits a cubic or quadratic model, then corrects and can refine with a Brent variant. `IBRION=3` integrates a damped second-order equation; negative `SMASS` activates velocity quenching. `IBRION=1` RMM-DIIS uses history and is sensitive to inaccurate forces. | CG variant 2 uses line search; variant 1 uses nested CG. BFGS and L-BFGS are quasi-Newton; SD is steepest descent; CG-BFGS switches when `relax_cg_thr` is reached. |
| History/reset | Public input/source confirms the `vc-relax` BFGS route and convergence scalars, but does not document every secant rejection/reset condition. Do not infer these from the keyword names. | Public structure-optimization text states that linearly dependent old vectors are removed from the RMM-DIIS history and that far-from-minimum history can be harmful. It does not expose the full cell-history reset state. | Public docs distinguish algorithm variants but do not specify all history reset/rejection rules. Treat those rules as unknown unless a versioned source function is inspected. |
| Convergence | `etot_conv_thr`, `forc_conv_thr`, and `press_conv_thr`; QE says the pressure threshold applies in addition to ionic criteria. `press_conv_thr` default is 0.5 kbar in the 7.5 input page. | `EDIFFG<0` is documented as a threshold on every force norm, while positive values test the energy change. The optimization output separately reports `g(F)` and `g(S)`. This documentation alone does not establish a portable absolute stress threshold or the complete cell stopping Boolean. | Public input keywords include force and stress convergence thresholds and `relax_nmax`; exact combined Boolean/norm semantics should be checked against the selected release. |
| Symmetry and numerical error | `ibrav`/`cell_dofree` can make some combinations mathematically unable to converge. SCF accuracy and stress evaluation remain part of the oracle contract. | Default symmetry can make lower-symmetry structures inaccessible. VASP warns that the PAW basis is not adjusted as the cell changes; volume relaxation therefore needs generous `ENCUT`/`PREC` to control Pulay stress. | `symmetry`, fixed-cell restrictions and implementation variant can change the accessible state space. The public optimization page does not imply physical validation merely from an optimizer status. |

## Coordinate, stress and scaling implications

QE's public interface uses lattice vectors and fractional/Cartesian input representations, while VASP and ABACUS also expose lattice vectors plus fractional or Cartesian positions. None of the three manuals gives a portable promise that their optimizer's internal cell variables are the same as a six-component symmetric strain vector. In particular, a raw nine-component cell matrix contains a global rotation gauge; a symmetric six-component chart removes that redundancy only after a reference-frame convention is fixed.

For ASE, stress is tensile-positive and `get_stress()` is in eV/Å³. With positive compression pressure (p), the physical differential used by this repository is

\[
d(E+pV)=-F:dR_{\rm nonaffine}+V(\sigma_{ASE}+pI):A,
\qquad A=L^{-1}dL,\quad dR_{\rm nonaffine}=dR-RA .
\]

Here lattice vectors and atomic positions are rows, and dR is the total Cartesian displacement. The nonaffine term is essential: the reported stress already includes the energy response to homogeneous affine motion of the atoms. Including that motion again in the force work would double count it. This is the contract implemented in `vc_geometry.py`; a VASP stress must be sign-converted before use. QE's `press`/stress units are kbar, and VASP commonly prints kB, so unit conversion is part of the adapter rather than an optimizer parameter. The scalar pressure term is not a replacement for the full deviatoric stress: hydrostatic cancellation can leave shear residuals.

The repository's map

\[
H=H_0\exp S,\qquad R=X\exp S,\qquad q=(\operatorname{vec}X,L_s s_6)
\]

uses an orthonormal symmetric basis and an explicit `strain_length` (L_s) in Å. It is a search metric, not a physical rescaling of (E+pV). The Frechet derivative of the matrix exponential is required at finite strain; replacing it with (dS) is only a small-strain approximation. `block_ssw.py` and `generalized_numerics.py` consequently need a flat coordinate space with a declared norm; a scale based on N or volume therefore requires a declared metric argument; a force threshold alone cannot determine it. For example, the Euclidean norm of affine displacement grows with the number and spatial distribution of atoms, but this does not by itself prescribe a universal scale for a periodic lifted chart.

## Why cell relaxation can fail

1. **Oracle error:** incomplete SCF, noisy forces, inconsistent force/stress evaluation, finite-basis Pulay stress, or changing k-point/basis conventions. VASP explicitly documents Pulay stress for changing-volume calculations; this is a systematic derivative error, not a condition number.
2. **Physical soft modes:** small elastic eigenvalues, molecular translations/rotations, weak adsorption and near-degenerate structures produce long steps and poor secant information. A large condition number is a useful diagnosis, not a root cause that identifies which coordinate or oracle is wrong.
3. **Coordinate redundancy:** raw cell matrices include rotation; periodic atoms include global translation; symmetry can remove physically relevant lower-symmetry branches. These are rank/null-space issues and can look like ill-conditioning.
4. **Mixed scales and coupling:** atomic forces (eV/Å) and cell derivatives (eV/Å in a chosen chart) are coupled through internal relaxation. Alternating minimization can zig-zag when the cross Hessian is large; joint optimization can be unstable when the metric is badly scaled.
5. **Nonconvexity and history:** BFGS/DIIS secants collected far from a basin or across a changing chart can be misleading. Line searches, curvature rejection and history resets are algorithmic safeguards, not evidence that the PES itself is ill-conditioned.

## Local block-Hessian interpretation (project inference)

At a smooth, locally stable minimum, after removing exact zero modes and applying the active constraints, write the Hessian of the declared objective in atomic coordinates x and strain coordinates s as

\[
\mathcal H=\begin{pmatrix}H_{xx}&H_{xs}\\H_{sx}&H_{ss}\end{pmatrix}.
\]

If Hxx is invertible on that active space, stationarity of the relaxed atoms implies dx/ds = -Hxx^{-1}Hxs. The curvature seen by an exact inner-atom minimization followed by a cell update is therefore

\[
H_{ss}^{\rm relaxed}=H_{ss}-H_{sx}H_{xx}^{-1}H_{xs}.
\]

Thus a crystal can have a stiff clamped-ion cell response but a soft relaxed-ion response. Alternating and joint schemes are handling this same coupling differently; “cell stiffness” alone is not enough to choose an optimizer. This is a local elimination argument, not proof that a joint scheme is universally faster. At a biased escape instability the Hessian can become indefinite or singular, invalidating the stable-minimum assumptions.

Condition number is meaningful only after specifying coordinates and metric. With q = D z, the Hessian becomes D^{-T} H D^{-1}; rescaling cell coordinates can change its condition number without changing the physical objective. A coupled metric test must separate this numerical effect from SCF errors and from the actual soft relaxed-ion modes. None of these facts justifies adding an uncalibrated cell mass or arbitrary restart threshold to Safe-total.

## Minimal, falsifiable design priorities for independent ASE VC-SSW/Safe-total

**Priority 1 — reuse existing derivative evidence.** `tests/standalone/test_vc_geometry.py` already checks real Cu EMT finite triclinic strain, all six strain derivatives, nonzero pressure, and exact metric rescaling; `test_vc_ls.py` extends the pullback check to LS. These establish implementation consistency, not search efficiency. Rerun or extend them only when a coordinate, derivative or pressure implementation actually changes; do not repeat a completed diagnostic as a new research campaign.

**Priority 2 — metric isolation, now source-motivated.** ABACUS cg2 explicitly uses `fac_stress=fac_force/N` and cell gradient inner products weighted by `1/N` (see the companion [pinned source notes](2026-09-12-dft-cell-relax-source-details.md)). For replicated material at fixed density, a homogeneous-strain energy derivative grows with N, whereas a useful strain increment should remain intensive. In our chart q_s=L_s s, a scalar initial inverse Hessian gives a strain step proportional to the strain gradient divided by L_s squared. Therefore L_s squared proportional to N is a principled candidate *for that initial metric*, with a separate dimensional material scale still required. This is not a universal optimum, and scaling the chart also changes the meaning of any Gaussian width in that chart. First isolate it in true quench; do not silently change the bias geometry. Hold seed, oracle, budget, active six cell DOF and stopping certificate fixed. Compare the existing explicit `strain_length` chart against one physically justified alternative scale, with no change to SSW bias or optimizer. The falsifiable hypothesis is that a scale that equalizes typical atomic and cell displacement magnitudes reduces line-search rejection/zig-zag without changing certified minima at equal oracle cost. If it does not, do not add adaptive scaling.

**Priority 3 — coupling isolation.** Compare `vc_reference.py`'s joint Safe-total route with the existing fixed-cell preparation followed by cell relaxation, using the same fresh E/F/stress certificate. The hypothesis is conditional: strong atom-cell cross coupling should favor joint relaxation; weak coupling should make alternating relaxation cheaper or equally reliable. Report failures, requests and certified outcomes, not acceptance rate alone.

**Priority 4 — numerical-error separation.** Repeat only the derivative/landing diagnostic with deliberately tighter versus ordinary oracle tolerances on the same toy/real ASE backend. A change in convergence under oracle tightening supports numerical noise/Pulay-like error; unchanged slow modes support physical softness or metric/coupling. This is a diagnostic, not permission to launch DFT jobs.

Keep fixed: no unexplained `N` scaling, no hidden pressure/stress threshold conversion, no cell-chart rebase while Gaussian history is live, no replacement of the physical landing certificate by inner optimizer status, and no claim that an ASE implementation reproduces QE/VASP/ABACUS native trajectories.

## Source record and unknowns

The versioned sources consulted on 2026-09-12 are:

- QE 7.5 official input reference: [INPUT_PW.html](https://www.quantum-espresso.org/Doc/INPUT_PW.html), especially `cell_dynamics`, `press`, `press_conv_thr`, and `cell_dofree`.
- QE public source, develop branch: [`PW/src/input.f90`](https://gitlab.com/QEF/q-e/-/blob/master/PW/src/input.f90), which maps `vc-relax` plus `cell_dynamics='bfgs'` to the variable-cell BFGS path and maps `damp-pr`/`damp-w` to extended-Lagrangian routes.
- ABACUS public optimization documentation: [`docs/advanced/opt.md`](https://github.com/deepmodeling/abacus-develop/blob/develop/docs/advanced/opt.md), including CG 1/2, BFGS 1/2, L-BFGS, SD, fixed DOF and line-search descriptions. The page is a develop-branch document; exact release behavior must be pinned for a benchmark.
- VASP public wiki: [`ISIF`](https://vasp.at/wiki/ISIF) for stress sign, cell DOF and Pulay-stress warning; [`Structure optimization`](https://vasp.at/wiki/index.php/Structure_optimization) for CG trial/corrector, RMM-DIIS history, damped dynamics and reported force/stress norms; [`POTIM`](https://vasp.at/wiki/POTIM) for step-width scaling; [`Volume relaxation`](https://vasp.at/wiki/Volume_relaxation) for basis-set and Pulay-stress guidance.
- VASP [`EDIFFG`](https://vasp.at/wiki/index.php/EDIFFG) documents the energy/force stopping rule; it is insufficient evidence for a universal cell stress tolerance.
- VASP source is proprietary. No VASP source function, internal cell coordinate map, exact BFGS reset rule, or patch-specific stopping Boolean is asserted here. Public wiki behavior is documentation evidence, not an inspected source implementation.

The native PAM-SSW convergence notes in `docs/research/native-vc-convergence-contract.md` are deliberately not used as evidence for QE/VASP/ABACUS internals. They reinforce the same boundary: a scalar pressure diagnostic is not a full residual-stress certificate, and a numerical optimizer stop is not a physical landing certificate.

Detailed QE/ABACUS source findings and immutable file links are in [the companion source audit](2026-09-12-dft-cell-relax-source-details.md). This follow-up motivates a future metric experiment; it does not change the current fixed-cell/constraint implementation priority.
