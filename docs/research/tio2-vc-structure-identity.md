# TiO2 joint-VC structure identity: no reference match established

2026-09-10. **None of the six pairs among rutile initial, its joint-VC landing, cell-quenched anatase reference and cell-quenched TiO2-B reference matched in the tested tolerance sweep.** This includes initial versus landing. The landing is an unassigned structural candidate; it is not assigned a new phase name. This analysis made **zero E/F/stress calls** and did not relax structures.

The original, restricted audit used the current local pymatgen `StructureMatcher` API and source were inspected in `mace_env`, with `PYTHONNOUSERSITE=1`. We explicitly set `scale=False`, `primitive_cell=False`, `attempt_supercell=False`, `allow_subset=False`, and `SpeciesComparator`. All inputs have the same Ti4O8 composition and 12 sites. No atoms are omitted and no volume rescaling hides density changes. Equivalent lattice bases, rotations, translations and same-species site assignment are handled by the matcher. This is a same-size-cell audit; it does not exhaust other supercells or a relaxed-path correspondence.

Before comparing different structures, each of the four inputs passed three controls at nominal site tolerance 0.002 Å, fractional lattice tolerance 0.001 and angle tolerance 0.1 degree: an exact copy; a combination of determinant-one integer lattice-basis change, rigid rotation, translation and site permutation; and Cartesian coordinate noise bounded by 0.0001 Å per site. Thus **12/12 controls passed**. These are numerical invariance/noise controls, not repeat physical quenches; they do not establish a physically universal structure threshold.

The predeclared sweep used nominal site distances / fractional lattice tolerances / angles:

| Nominal distance (Å) | Fractional lattice tolerance | Angle tolerance (degrees) | Matches among six pairs |
|---:|---:|---:|---:|
| 0.002 | 0.001 | 0.1 | 0 |
| 0.02 | 0.005 | 0.5 | 0 |
| 0.05 | 0.02 | 2 | 0 |
| 0.10 | 0.05 | 5 | 0 |

`stol` is dimensionless. We convert the nominal distance d by `stol=d/ell`, `ell=((V1+V2)/(2N))^(1/3)`. The inspected matcher internally normalizes by each candidate's averaged lattice volume, so these are **nominal** Å tolerances, not exact uniform Cartesian hard cutoffs. The saved dimensionless stol, fractional lattice tolerance and angular tolerance are the authoritative settings. With scale disabled, lattice/density differences can reject a pair independently of site matching; an unmatched result does not supply a Cartesian RMS distance. We do not report a fabricated RMS for failures.

## Density and Ti–O shells corroborate differences

The shell entries below are mean sorted distances across the four Ti sites, using periodic O images within 4 Å. No bond threshold is chosen. A large gap after neighbor six is visible in each structure; these distances support distorted six-neighbor environments but do not uniquely identify a polymorph.

| Structure | Volume (Å³/12 atoms) | Mean ordered Ti–O neighbors 1–6 (Å) | Neighbors 7–8 (Å) |
|---|---:|---|---|
| Rutile initial | 127.6213 | 1.9414, 1.9432, 1.9460, 1.9845, 1.9869, 2.0419 | 3.4652, 3.5037 |
| Joint-VC landing | 125.5703 | 1.8793, 1.8811, 1.9633, 1.9641, 2.1173, 2.1214 | 3.2072, 3.2090 |
| Anatase reference | 139.0273 | 1.9550, 1.9550, 1.9550, 1.9551, 1.9786, 1.9786 | 3.8636, 3.8637 |
| TiO2-B reference | 146.0356 | 1.8213, 1.8899, 1.9513, 1.9513, 2.0855, 2.2530 | 3.6567, 3.6567 |

The landing shrinks volume by about 1.61% relative to the saved initial and changes the local Ti–O distortion. It remains above the separately prepared anatase and TiO2-B reference energies on this MACE PES. Together with [positive sampled joint curvatures](rutile-joint-vc-hessian-qualification.md), this supports retaining it as a distinct-at-tested-tolerances, lower-energy-than-initial candidate. Finite residual forces, possible tighter-quench changes, tolerance dependence, and restricted reference coverage prevent declaring an exact basin identity or new material phase. The supplied rutile name identifies source provenance, not an independently proved final symmetry assignment.

Reproducer: `research/ga_ssw/match_tio2_vc_structures.py`. Evidence: `research/ga_ssw/evidence/tio2-vc-structure-identity/` includes local matcher API/source excerpt, code, versions, checksummed source result paths, exact arrays, all 12 controls, 24 pair/tolerance results, and per-site neighbor distances. Subsequent tighter physical quenches would be a separately authorized and separately costed task, not something performed by this matching report.


## Representation-equivalence recheck after Cu4 calibration

A subsequent Cu4 calibration showed that disabling primitive reduction can reject equivalent repeated-cell representations even after geometric tolerances are enlarged. Therefore the original TiO2 negative matches alone were insufficient to claim distinct crystal identities. The original evidence directory is preserved unchanged as the restricted-matcher result.

Re-ran **all six TiO2 pairs at all four original tolerance settings**, now with `primitive_cell=True`, `attempt_supercell=True`, `scale=False`, `allow_subset=False`, and the same species comparator. All **24 comparisons still report no match**. All **12 original duplicate/transformation/noise controls pass** again. This recheck used zero E/F/stress calls; no coordinates, volumes or physical states changed.

Thus no reference assignment changes in this broader representation-aware check: the landing remains an unassigned candidate, including with respect to its initial structure. A negative match remains bounded by the tested matcher, tolerance, reference set and residual relaxation error; it is not itself proof of a new structure or phase. The prior sentence describing a candidate as distinct at tested tolerances refers only to these explicit comparison outcomes.

New reproducer: `research/ga_ssw/rematch_tio2_vc_primitive.py`. New evidence: `research/ga_ssw/evidence/tio2-vc-structure-identity-primitive/`, preserving the full configuration, 24 pair results and 12 controls separately from the original restricted audit. This updates the original method limitation without overwriting its historical evidence.
