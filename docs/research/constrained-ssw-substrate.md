# Fixed-substrate SSW and an actual Cu(111) adsorption-site change

2026-09-10. `pamssw/standalone/constrained_reference.py` implements complete fixed-cell, fixed-atom SSW. This supports the numerical substrate constraint needed by TYPE4 surface exploration; it is not native surface-GA parity, a general arbitrary-constraint solver or variable-cell constrained SSW.

## Geometry and certificate semantics

`ReducedCartesianChart(atoms, fixed_indices=...)` freezes the listed atom positions and the cell. If indices are omitted, it reads the union of ASE `FixAtoms` constraints. Explicit indices must agree with any existing `FixAtoms`; other constraint classes are rejected. At least one fixed and one active atom are required. Full, partial and no PBC are supported. Only active Cartesian displacements are coordinates, in Å: `R_active=Rref_active+q`, `R_fixed=Rref_fixed`. The gradient is exactly `g_q=-F_raw_active`. There is **no** global translation/rotation projection: relative motion against the fixed substrate is physical.

Calculator Atoms have their ASE constraints removed so the existing unmodified `ASESurface` returns raw physical E/F, including forces on fixed atoms. Fixed coordinates remain exact by construction. `constrained_quench` minimizes the true PES on this same manifold using Safe-total and performs a fresh raw E/F evaluation. Its certificate explicitly reports:

- `scope='fixed_atom_manifold'`;
- maximum active force, `active_fmax`;
- maximum raw force over all atoms, `full_raw_fmax`;
- exact fixed-coordinate/cell invariance.

Convergence requires the optimizer and the active-force/invariance certificate. A nonzero fixed-atom force is an external support reaction, not automatically a failure of this constrained problem. It also means the result is not an unconstrained stationary structure. Returned candidate Atoms retain `FixAtoms` so downstream users do not silently forget the manifold. Independent raw-force checks must remove the copied constraints before using `ASESurface`, as the probe does. Input Atoms are not mutated.

## Complete independent workflow

```
config = ConstrainedSSWConfig(width=..., rotation_bias=...)
result = run_constrained_ssw(
    atoms, surface, fixed_indices=..., steps=..., config=config, rng=rng,
)
```

The private shared reduced-coordinate lifecycle now accepts an optional true-quench callback; existing isolated single-chain/forest behavior is unchanged. This driver supplies the constrained true quench and active-displacement chart. Initial true relaxation, dimer direction softening, a fixed initial anchor throughout climbing, accumulated conservative Gaussian terms, Safe-total biased relaxation, final true constrained quench and one Metropolis decision are all executable. Each proposal freezes its reference while its bias history is live. Bare energy determines MC at fixed cell. Certified rejected candidates remain visible; failed stages preserve current and all request costs. Full fixed coordinates and the cell are checked against the initial reference at each chart rebuild.

The config exposes the same numerical/search settings as the existing independent kernel, without unused RC angular metric parameters or new heuristics. Width/max_step are Å, rotation bias eV/Å², force tolerances eV/Å and temperature K. These are not all asserted universal paper defaults. There is no LS, VC, topology inference or general constraint Jacobian in this subset.

## Tests and real Cu surface probe

Five new tests cover active-gradient finite differences, exact fixed geometry and preserved PBC, unsupported/inconsistent constraints, active certification with a deliberately large raw fixed force, complete Safe/dimer/Gaussian/MC wiring, and failed proposal cost/current retention. Together with existing RC/forest/VC tests: **24 passed**, existing ASE/NumPy deprecation warnings only. The analytic tests establish numerical contracts, not scientific effectiveness.

`research/ga_ssw/probe_constrained_cu111_emt.py` constructs an actual ASE `fcc111('Cu', size=(2,2,3), a=3.6, vacuum=8)` slab and a Cu adatom at the supplied FCC site, height2 Å. The lower two layers (8 atoms) are fixed; four top-layer atoms and the adatom remain active. PBC is `(True,True,False)`. EMT is an approximate metallic Cu PES appropriate for a bounded Cu-surface demonstration; its results are not DFT predictions.

The predeclared seed3, one-outer-step probe used width .2 Å, rotation bias100 eV/Å², at most two Gaussians, HVP limit20, quench limit200, active fmax .01 eV/Å and the unchanged core settings. These are development settings, with no parameter tuning or claimed optimality. The hard ceiling was500 E/F and30 s on one CPU thread. Observed cost was **60 search + 2 independent-calculator fresh calls = 62 E/F, 0.115 s**. Initial quench spent8; the proposal/final quench spent52, including50 in a single converged Gaussian stage. The lower true energy triggered completion before the two-Gaussian maximum. All62 request rows reconcile.

| Fresh observable | Initial constrained candidate | Final constrained candidate |
|---|---:|---:|
| Energy (eV) | 3.549170396080 | 3.543911464044 |
| Active maximum force (eV/Å) | 0.000833170 | 0.002701438 |
| Full raw maximum force (eV/Å) | 0.184951355 | 0.108235189 |
| Adatom height over mean top-layer z (Å) | 1.908755 | 1.908322 |
| Nearest adatom–substrate distance (Å, MIC) | 2.424681 | 2.423977 |

The fixed eight atoms and cell are exactly unchanged in both independent checks. Fresh energies agree within1.6e-14 eV. The candidate was accepted with ΔE=-0.005258932 eV. Its adatom moved laterally about1.468 Å while remaining adsorbed, rather than translating the whole system.

`analyze_constrained_cu111_registry.py` performs a **zero-E/F** geometric comparison to the explicit FCC/HCP positions supplied in ASE's `adsorbate_info` for this Cu(111) lattice, including primitive-cell translations. The initial adatom is within2e-15 Å of an FCC site. The final adatom is0.001667 Å from an HCP site and1.468150 Å from the nearest FCC site. This supports a concrete FCC-to-HCP adsorption registry change relative to the fixed substrate. It does not certify Hessian stability, an elementary transition path, barrier, physical diffusion rate or repeated-seed efficiency. No other seed or competing method was run.

Complete input/config/code snapshots, all E/F/geometry calls, results and the zero-cost site analysis are retained in `research/ga_ssw/evidence/constrained-cu111-emt/`. The frozen probe snapshot precedes a subsequent pre-oracle bounds check for malformed `FixAtoms` indices; that input-validation change does not alter this valid geometry or numerical trajectory.
