# Executable independent VC-RC SSW workflow

2026-09-10. `pamssw/standalone/rc_vc_reference.py` now connects the exact periodic rigid-forest geometry to dimer softening, conservative Gaussian climbing, full all-DOF true quench and enthalpy MC. This closes the **independent executable workflow subset**, not native Kabsch/lambda or complete RC release parity.

## Public contract and lifecycle

```
config = RCVCSSWConfig(
    rotation_length=..., torsion_length=..., strain_length=...,
    width=..., rotation_bias=..., pressure=...,
)
result = run_rc_vc_ssw(
    atoms, surface, trees=components, anchor=0,
    steps=..., config=config, rng=rng,
)
```

`surface` is an `ASEStressSurface` or equivalent counted E/F/full-stress oracle. Required topology and metric semantics are documented in `rc-vc-geometry-contract.md`. Coordinates must be an explicit consistent periodic lift of each component; no automatic wrapping, image reconstruction or chemical inference is performed. Calculator translation invariance is a physical precondition, not inferred from the presence of stress support.

1. Validate the full forest/cell domain before oracle work. Perform a **bare full atomic and cell** `cell_quench` for the initial structure, including its fresh full-force/residual-stress certificate.
2. For each outer proposal, build a new `RigidForestCellChart` from the selected initial/current true minimum. All root rotations, non-anchor relative translations, torsions and symmetric strain coordinates participate. Freeze this chart and its reference body geometry for the entire climb.
3. Draw one scaled-coordinate random anchor. Every dimer stage retains that same initial anchor. The dimer objective uses the exact physical enthalpy gradient, including nonaffine force/virial terms. Unconverged direction refinement is a failed proposal.
4. Add projected Gaussian terms to the physical enthalpy, with the same explicit forward-force height formula as the other independent kernels. Safe-total minimizes that combined scalar objective in the same scaled coordinates. No separate force projection or native lambda rescaling changes its gradient. Nonpositive heights and failed biased quenches remain failures.
5. At lower true enthalpy or the Gaussian limit, remove all bias **and release all rigid constraints**, then call shared `cell_quench` on the physical E+pV surface over all Cartesian atomic and allowed cell DOFs. Require both the optimizer result and the independent physical force/stress certificate. One MC decision uses `delta(E+pV)`, not energy alone. Record all certified landings including rejected ones; current changes only on acceptance.

The final unrestricted quench can alter intramolecular lengths/angles, so the next accepted chart freezes that new body geometry. Topology indices persist; preserving their chemical interpretation is a separate physical qualification requirement. The implementation does not infer a new rigid partition after a reaction or dissociation.

## Parameters, records and limits

`RCVCSSWConfig` extends the isolated forest settings with mandatory `strain_length` and explicit pressure/stress tolerance. Physical pressure and residual stress are eV/Å³, positive pressure means compression. The three metric lengths set coordinate geometry; width/max_step are lengths in that scaled chart. Proposal stopping uses reduced modified-gradient tolerance; final true stopping uses full physical force and stress thresholds. Those are distinct criteria.

`VCSSWResult` provides initial/current/best physical evaluations, all certified candidate minima, per-outer events and per-stage records. Events retain the full true-quench object and certificate. Costs count initialization, dimer finite differences, line searches, failed evaluators, physical checks, final quenches and certificates through the supplied surface request counter. A bounded oracle wrapper can impose an absolute cost/wall ceiling; step limits alone are not E/F/stress limits. No hidden backoff or automatic retuning is added.

Native center/cell Kabsch alignment, lambda force transmission, native angle-coordinate stream and input-controlled native policy remain unimplemented. The exact independently defined map is fully documented. Periodic lifting inference, linear/point roots, loop constraints and explicit angle-axis chart-rank rejection remain outside this subset. No molecular-crystal success or efficiency conclusion follows merely from workflow completion.

## Verification and actual ASE end-to-end wiring

`tests/standalone/test_rc_vc_reference.py` adds four lifecycle checks: full biased path/final certificates and fixed anchor; an MC case where physical energy decreases but enthalpy increases, proving selection uses the declared objective; initial-certificate failure; failed rotation cost/current preservation; and nonperiodic pre-oracle rejection. These tests use controlled surfaces/certificates to isolate wiring and are not material validation. Combined geometry, isolated RC, forest and VC-RC tests: **19 passed**, existing ASE/NumPy deprecation warnings only.

`research/ga_ssw/probe_rc_vc_cu4_emt.py` provides a separate complete actual-ASE test. Input is conventional FCC Cu4 at 3.6 Å, with the four atoms explicitly grouped as one artificial nonlinear rigid body during climbing. EMT is a metallic Cu potential; this grouping is deliberately a numerical wiring device, **not a molecular model**. No inappropriate periodic water potential or additional MACE campaign was substituted to manufacture a molecular-crystal claim.

The frozen development configuration uses seed3, one outer step, one Gaussian, Lrot=Ltor=2 Å/radian, Lcell=4 Å, width .2 Å, rotation bias100 eV/Å², HVP limit20, relax limit200, fmax .01 eV/Å and stress tolerance .001 eV/Å³ at zero pressure. These explicit values are not claimed paper defaults or tuned optima. The total ceiling was 400 E/F/stress requests and 30 seconds on one CPU thread.

Measured result: **35 search + 2 independent-calculator fresh checks = 37 E/F/stress requests, 0.132 seconds**. Initial quench spent4 requests; the proposal and final quench spent31, including18 in its converged biased stage. All37 JSONL rows reconcile. The biased cell developed nonzero shear while all six intrabody pair distances were unchanged to the serialized floating-point result (maximum difference0). The final full atomic/cell quench then restored a small-residual physical structure.

| Fresh measurement | Initial | Landing |
|---|---:|---:|
| Energy/enthalpy at p=0 (eV) | -0.0281458659312 | -0.0281263477456 |
| Maximum force (eV/Å) | 2.09e-13 | 0.00370944 |
| Maximum residual stress (eV/Å³) | 0.0000609281 | 0.000521743 |
| Volume (Å³) | 46.2581785 | 46.2668146 |

Fresh energies match stored energies exactly in this run. The +0.0000195182 eV candidate was MC accepted. Its small energy difference and near-return are not proof of a new basin or phase, and no Hessian or strict structure identity was evaluated. This establishes end-to-end numerical operation with an actual periodic ASE energy/force/stress backend; appropriate molecular-crystal physics and repeated-seed search efficiency remain unvalidated.

Complete input/config, code snapshots, every E/F/stress/geometry request and result: `research/ga_ssw/evidence/rc-vc-cu4-emt/`.
