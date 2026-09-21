# Fe7C3-80: normal biased numerical stops followed by physical quench

2026-09-11. The strict biased-gradient gate is unnecessarily restrictive for
proposal generation, but removing it alone has not improved low-energy search.

The independent joint log-strain workflow now has an explicit experimental
`bias_release=numerical_stop` option. Normal maxiter/native stopping at a finite
accepted biased point ends climbing and attempts a real E+pV quench. Oracle,
request-limit and abnormal native errors remain failures. Strict remains the
control default. Final physical fmax0.001 eV/A and stress0.0001 eV/A^3 are
unchanged. SciPy's native status classification was corrected to distinguish
successful numerical stopping, iteration limit and abnormal line-search exits.

## Controlled complete-workflow experiment

Artifacts: `research/ga_ssw/fe7c3-vc-numerical-release/`, including frozen source,
input, plan, manifest, per-request ledgers, results, independent fresh checks,
`audit-summary.json`, `identity-summary.json`, and `prefix-comparison.json`.
The qualified initial Fe7C3-80 structure and model are identical to the parent
`fe7c3-vc-mature-baselines`. Width0.6A, rotation bias100, NG14, memory10,
maxiter300, two outer attempts per arm, seeds7/101, pressure0, temperature300K,
strain length8.985367616360952A and all optimizer rules remain as recorded in
that parent. Each arm has 2000 EFS inclusive of up to3 fresh calls and480s.
No LS, parameter scan, cap extension or native schedule parity is implied.

Job1261354 completed exit0 on one V100/4v100n10 in4m06s.

| Optimizer | Seed | Total EFS | Search/fresh | Paid/requested attempts | Certified noninitial proposal endpoints |
| --- | ---: | ---: | --- | --- | ---: |
| Safe-total | 7 | 1999 | 1997/2 | 2/2 | 1 |
| Safe-total | 101 | 1998 | 1997/1 | 1/2 | 0 |
| SciPy L-BFGS-B | 7 | 325 | 323/2 | 2/2 | 1 |
| SciPy L-BFGS-B | 101 | 175 | 172/3 | 2/2 | 2 |
| ASE LBFGSLineSearch | 7 | 1998 | 1997/1 | 1/2 | 0 |
| ASE LBFGSLineSearch | 101 | 1998 | 1997/1 | 1/2 | 0 |

Total8493EFS =8483search+10fresh; 9paid/12requested attempts. All ledgers
reconcile. Initial structures are excluded from the last column. These are
proposal endpoints, not four distinct newly discovered minima.

Three SciPy endpoints match their initial structure and each other under all
three previously declared pymatgen tolerance profiles. Energy differences are
about1.05–1.48e-6eV. Safe-total's one endpoint is different under all three
profiles, higher by13.443783eV and MC rejected. No endpoint lowers initial
energy. No new endpoint Hessian was evaluated, so even the different stationary
candidate is not established as a stable phase. SciPy seed7's second physical
quench stopped with fmax0.00556eV/A and is correctly retained as a failure.

The parent strict experiment had no qualified noninitial endpoints. However,
these are not bitwise paired trajectories: first evaluations at identical
coordinates already differ at roughly1e-14eV/A in forces, and finite-difference
rotation amplifies small differences. In particular ASE seed7 now converges an
intermediate biased stage at step300 and continues until the request cap,
where the parent stopped earlier. Do not attribute every changed path or cost
solely to the release flag or infer an optimizer ranking from two seeds.

## Decision

Retain release as an experimental lifecycle option, not a promoted optimizer
or search default. The counterfactual and this full-budget experiment establish
that biased numerical stopping can precede valid physical quench; they do not
establish improved low-energy exploration. SciPy's cheaper repeated initial
structures are not superior coverage. Stop tuning joint biased thresholds as
the main reproduction task: next qualify the existing cell/atom block workflow
and resolve its cell-direction continuity and native force/coordinate masks.
The 2014 paper's staged schedule remains distinct from the joint implementation.

Registered Fe7C3 cost becomes56942EFS (prior48449+8493), before subsequent block
experiments. Cu/EMT regression costs are recorded separately; unit tests and
these short trajectories are not production-level global-search validation.
