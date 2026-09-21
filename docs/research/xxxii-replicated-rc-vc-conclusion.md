# XXXII: corrected periodic representation enables a complete RC-VC proposal

2026-09-11, experimental research worktree. This is a 172-atom, four-molecule
XXXII example with converted original GAFF topology. The ASE walker is Python;
LAMMPS supplies E/F/stress, with no LASP or Java execution in the search.

The previous primitive-cell representation changes nonbonded exclusions at a
componentwise half-box crossing on a sheared cell, creating a 0.234615 eV jump.
It cannot serve as an arbitrary-cell oracle. Explicit fixed 1×1×2 replication
preserves the infinite crystal and the 172-atom public ASE state; the internal
344-atom energies and corresponding forces are folded back to the primitive
representation. No force-field coefficient, pressure or molecular restraint
was changed. Topology-distance 1–3 pairs must remain inside the restricted
replicated half-box components; invalid geometry fails before engine work.
See [adapter design](xxxii-replicated-calculator-design.md) and
[discontinuity evidence](xxxii-periodic-exclusion-discontinuity.md).

33 real EFS checks cover the initial structure and two old failure-point
geometries in 1×1×2, 1×1×3 and 2×2×2 representations. Energy differences are at
most 2.2e-11 eV. Tested RC/cell directional derivatives differ by at most
1.6e-6 in the scaled chart units; the known erfc energy/force discrepancy
remains. These checks support a bounded experiment, not exact conservativity.

Both whole-step runs start afresh from the same original input, seed 3, with
the same 55-dimensional rigid-forest/cell chart, memory 400, 12 Gaussian limit,
5000 API / 120 s cap, and force/stress tolerances. The only numerical operator
change is forward two-vector dimer versus central-difference Ritz rotation.

| Arm | Total API, including fresh checks | Engine calls | Internal atom evaluations | Outcome |
|---|---:|---:|---:|---|
| Existing forward dimer | 1319 | 1305 | 448920 | Four biased stages converge; stage index 4 rotation fails at residual 0.06106 with 100 requests; no landing |
| Central Ritz | 3559 | 3533 | 1215352 | All 12 stages and unrestricted true cell quench converge; fresh candidate, MC rejected |

Wall times are 33.42 and 95.94 s, respectively, on the same bounded CPU runtime.
The completed candidate has E=-6.379435727368919 eV, 0.315744343448799 eV
above the initial minimum. Fresh fmax=0.00979260818729 eV/Å and maximum residual
stress=1.04023252381e-5 eV/Å³ pass the unchanged 0.01 and 0.001 thresholds.
Both endpoints were also recomputed from full-precision saved coordinates in
1×1×3 and 2×2×2: maximum energy/force/stress deviations are 2.12e-11 eV,
8.53e-12 eV/Å and 2.31e-13 eV/Å³. An earlier four-call extxyz round-trip check
is retained separately; its 6.3e-7 force difference comes from saved coordinate
precision, and all four calls remain charged.

Volume changes from 1959.44 to 1937.86 Å³; the cell undergoes substantial
shear. All 180 explicit bond lengths remain close to the initial values
(largest paired change 0.00786 Å). The image-resolved contact audit is saved
with the run. Fixed GAFF topology cannot establish chemical stability, and
no Hessian or finite-temperature stability conclusion is drawn.

Decision: retain the replicated adapter as an experimental XXXII backend and
retain central Ritz for further controlled numerical comparison. This is the
first complete complex XXXII RC-VC proposal in this diagnostic series. It is
not lower-energy discovery, proof of universal solver superiority, or original
LASP whole-engine parity. A subsequent frozen comparison at the identical
stage-4 failed geometry separates difference accuracy from subspace size:

| Solver at identical q/anchor | Actual E/g requests, cap 100 | Direct residual | Converged at 0.02 |
|---|---:|---:|---|
| Forward two-vector dimer | 100 | 0.06106269 | No |
| Central two-vector dimer | 98 | 0.43354892 | No |
| Central Ritz | 26 | 0.00472644 | Yes |

Central differences reduce the two-vector projected asymmetry, but halving the
available HVPs under the same oracle budget leaves a worse residual here.
Thus merely replacing forward differences by central differences is insufficient
at this budget. The retained Ritz subspace matters in this measured case. This
does not establish failure of a sufficiently long central-dimer calculation.
The three arms cost 224 additional EFS / 77056 internal atom evaluations.
Their source, exact q/anchor and calls are frozen under
`xxxii-rc-vc-replicated-forward-control/frozen-rotation-comparison/`.

For terminology, [ASE's primary dimer source](https://docs.ase-lib.org/_modules/ase/mep/dimer.html)
uses `use_central_forces=True` to extrapolate the second endpoint from the center
and first endpoint (lines 502–520); it does not request both endpoints. The
central-difference probe here explicitly evaluates both q±h*n and is a controlled
modification of our two-vector solver, not a test of ASE's entire dimer optimizer.
The measured Ritz gain motivates a public opt-in solver interface; defaults stay
unchanged pending broader matched-system evidence.

The implementation is now independently packaged under `pamssw.standalone`:

```python
from pamssw.standalone import generalized_central_ritz, run_rc_vc_ssw

result = run_rc_vc_ssw(
    atoms, surface, trees=trees, anchor=0, steps=1, config=config, rng=rng,
    direction_solver=generalized_central_ritz,
    rotation_force_calls=100,
)
```

`run_vc_ssw` accepts the same two optional keywords. `surface` remains the
ordinary E/F/stress interface backed by an ASE calculator; no research-package
or binary solver dependency is introduced. The100-request value belongs to
this comparison and is not a new universal default. Omitting both keywords
preserves the existing dimer/HVP-budget path.

Public integration was checked against the full saved3557-request real search:
all requested atomic positions and cells agree exactly, as do final energy
and MC rejection (`public-api-replay-original-input`). The first replay attempt
incorrectly started from the first *evaluated* geometry rather than the original
input and accumulated an extra coordinate-roundtrip error; it is retained.
No tolerance was loosened to obtain the successful replay and no new PES calls
were made by either replay.

Two independent full-cell endpoint refinements (10× stricter force/stress
thresholds) cost542 additional API calls, including fresh independent engines.
Initial/candidate energies become -6.697158970679295/-6.384017021303422 eV,
so the +0.313141949375873 eV difference persists. Fresh fmax values are
0.00097154/0.00089983 eV/Å; residual stresses are4.88e-7/3.11e-7 eV/Å³.
The translation-projected joint atomic/cell Hessian qualification is now
complete at both spacings (1e-4 and5e-5 Å in the declared scaled chart).
All519 eigenvalues of the symmetric part are positive for each endpoint and
each spacing. At the smaller spacing:

| Endpoint | Smallest eigenvalue | Largest eigenvalue | Condition number |
|---|---:|---:|---:|
| Refined initial | 0.00281163 | 217.33076 | 77297 |
| Refined candidate | 0.000287395 | 213.57842 | 743152 |

Eigenvalues use eV/Å² in the explicit strain-length5 Å chart. The two-spacing
symmetric-matrix differences have operator norms1.62e-4/7.54e-5; the smaller-step
antisymmetric parts have operator norms6.60e-5/1.57e-5. These are measured
discretization/nonconservativity diagnostics, not a rigorous total error bound.
Both spectra are consistent with local finite-cell stability at the stated
stationarity tolerances. They do not prove all-wave-vector phonon stability,
chemical model validity, DFT stability, or equilibrium populations.

The first90-second-per-endpoint allocation ended with saved partial matrices
after1850 EFS. A completion run retained every completed column, checked one
overlapping column per endpoint (exact agreement), and added2310 EFS including
fresh checks. Total Hessian qualification cost4160 EFS, not2310; all raw
matrices remain in the original and completion directories. The I/O microcheck
did not support the provisional storage-overhead hypothesis; the wall
allocation was extended using the observed rate without changing the method.
See `xxxii-replicated-hessian-completion/root-spectrum-review.json`.

Evidence under `research/ga_ssw/evidence/`:

- `xxxii-replicated-qualification/{result,root-comparison}.json`
- `xxxii-rc-vc-replicated-completion/{plan,result,environment}.json`, `ledger.jsonl`
- `xxxii-rc-vc-replicated-completion/cross-representation-fullprecision/`
- `xxxii-rc-vc-replicated-completion/offline-geometry-audit.{json,md}`
- `xxxii-rc-vc-replicated-forward-control/`
- Full listed-series accounting: `xxxii-rc-vc-one-step/current-cost-ledger.json`

The strict endpoint identity audit (`xxxii-endpoint-identity-audit`) retains all
180 explicit bonds with maximum absolute length change0.007661 Å. Volumes
1960.9235→1932.2869 Å³ differ by1.46%. Periodic StructureMatcher with species
comparison, no volume scaling and three recorded diagnostic tolerance sets
returns no match in all three cases. This supports distinct model numerical
minima alongside the endpoint energy and curvature evidence, without identifying
a stable polymorph or a kinetic transition.

The28 reported joint dihedrals use actual bonded four-atom paths, checked
against every graph edge, and ASE MIC dihedrals. An earlier audit selected
invalid quadruplets for24 joints; it was corrected and retained under
`before-joint-path-review`. Corrected labeled angle changes range from
−18.34° to169.66°. A large labeled torsion difference alone is not proof of
a different molecular conformer because chemically equivalent atom
permutations have not been minimized over. None of this offline audit adds
PES calls.
