# Independent Cu4 joint-VC minimum and geometry checks

2026-09-10. Eight archived minima from two three-step walks were re-examined under a predeclared 1,500-request CPU cap. **All eight reach strict force/stress stationarity and have positive 15-dimensional joint atomic/strain Hessians at both finite-difference lengths. Geometry fingerprints distinguish two groups: the two initial states, and all six subsequent landings.** Total cost: **538 combined EMT E/F/stress requests**. No walk was rerun, no original success count changed, and no production code changed.

Sources: `research/ga_ssw/evidence/joint-vc-cu4-physical-stop/` and `joint-vc-cu4-l36-seed17/`, four minima each. Evidence: `research/ga_ssw/evidence/vc-cu4-minima-qualification/` contains the frozen plan, eight full per-case files, aggregate result, execution log and script snapshots. The final aggregate initially encountered a NumPy boolean JSON serialization error after every calculation had finished; aggregation was repaired from the eight saved files with **zero extra E/F/stress calls**. The corrected rerun script is `research/ga_ssw/qualify_vc_cu4_minima.py`.

Each landing underwent standard Safe-total true-PES quenching in its own reference chart with strain_length=3.6 Å, physical fmax≤1e-4 eV/Å and max absolute stress≤1e-5 eV/Å³. At most101 requests and100 steps were allowed per quench. All converged; an additional final E/F/stress evaluation was included per case. Maximum final force was 9.80e-5 eV/Å, maximum stress7.86e-6 eV/Å³.

## Structure discrimination before assigning phase labels

Fingerprints include all **species-resolved periodic neighbor distances within5 Å**, their per-atom lists, and cell volume; lattice periodic images are included. Pairwise tolerance was1e-3 Å and0.01 Å³. This is not energy-only matching.

Representative shells, coordination averaged per atom:

| Representative | Volume Å³ / four atoms | Neighbor shells: radius Å × count |
|---|---:|---|
| Initial group | 46.261536 | 2.538390×12; 3.589826×6; 4.396620×24 |
| Subsequent group | 46.245854 | 2.537571×6; 2.538617×6; 3.589406×6; 4.143011×2; 4.396403×18; 4.858904×12 |

The distinguishing shell populations persist beyond the near-identical first-neighbor shell. Initial energy is -0.0281459682 eV/cell; subsequent strict energies cluster around -0.03190653 eV/cell. These energy differences support the measured data but are not the identity criterion. All pairwise comparisons within each reported group pass the declared distance/volume tolerance, and comparisons across groups fail.

This evidence supports **two distinguishable local environments at the fingerprint resolution**, with repeated visits within each group. Radial fingerprints are not injective: homometric structures, topology and full lattice/atom equivalence are not resolved. No unique space group, crystal phase or absolute global minimum is assigned. Cell conditioning is representation dependent and was recorded without interpreting it as a physical stability certificate.

## Joint small-cell Hessian

At each strict endpoint a fresh local symmetric-log-strain chart was constructed. Its18 coordinates are12 atomic plus6 symmetric strain components; orthonormal projection removes three uniform atomic translations, leaving15 directions. Central gradient differences used h=1e-4 and5e-5 Å in the scaled chart,60 total E/F/stress calls per structure. Matrices were symmetrized after recording their antisymmetry norms.

Every eigenvalue is positive at both h values. Lowest eigenvalue is1.2735483 eV/Å² for initial states and1.59684–1.59708 eV/Å² for subsequent states, in the declared scaled chart. Maximum two-h eigenvalue difference is3.75e-7 eV/Å²; maximum matrix antisymmetry norm is6.50e-7 eV/Å². These are finite-difference and finite-cell local-minimum checks, not universal error bounds.

Positive curvature establishes stability against the tested **four-atom-cell atomic modes and homogeneous strains** under EMT. It does not test larger-cell or finite-wavevector phonons, defects, competing compositions, DFT stability or finite-temperature behavior. Consequently these results strengthen the existing small-cell VC demonstration without promoting it to general crystal-structure-prediction validation.

## Explicit hcp stacking reference

A subsequently authorized reference was built exactly with `ase.build.bulk('Cu','hcp',a=2.55,c=4.16).repeat((2,1,1))`, four atoms. One standard joint Safe-total quench used the same EMT, strain_length3.6, physical fmax1e-4 and stress1e-5, with150 total requests permitted. It converged in5 steps; including its independent final certificate, **7 E/F/stress requests** were used. Final energy is -0.0319065333 eV/cell, volume11.56145729 Å³/atom, fmax1.17e-13 eV/Å, max stress1.09e-7 eV/Å³.

All six subsequent VC landings match this reference's species-resolved periodic distance list within5 Å and volume under the predeclared tolerances. Their largest distance discrepancy is5.22e-5 Å and largest volume-per-atom discrepancy6.27e-5 Å³. Both initial fcc inputs fail that hcp fingerprint comparison; their initial source was already an explicitly constructed fcc cell.

Thus the observed second minimum group is **consistent with hcp stacking under this small-cell EMT comparison**. This is more specific than an energy-only assignment, but remains a finite-radius fingerprint match rather than complete atom/lattice isomorphism proof. It does not establish that hcp is the ground state of experimental Cu, that larger-cell phonons are stable, or that a general VC search benchmark has been passed. The known reference source and initial cell/positions are preserved in `research/ga_ssw/evidence/vc-cu4-hcp-reference/`; script `research/ga_ssw/compare_vc_cu_hcp_reference.py`. Combined independent qualification plus reference cost is545 requests.
