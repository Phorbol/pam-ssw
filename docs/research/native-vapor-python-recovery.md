# Independent cluster geometry recovered from native instructions

2026-09-11. Research implementation:
`research/ga_ssw/native_vapor_reference.py`. This file imports NumPy only.
It does not run LASP, request forces or assume a particular ASE calculator.

## Recovered finite isolated-coordinate algorithm

1. Sum each atom's Euclidean distances to all atoms and choose a minimum-score
   atom (a geometric medoid). Start with its `distance < criterion` connected
   component. Original loops use an in-place ordering and breadth-first expansion.
2. Detection mode reports the nearest distance between this initial component
   and all remaining atoms. It is **not** the closest pair of arbitrary components.
3. Repair mode chooses the nearest remaining atom to the growing anchor group,
   finds its still-unattached connected component and translates that component
   rigidly so the selected separation becomes `0.7 * criterion`. It then adds
   those members to the anchor group and repeats. No rotations, PES evaluations
   or acceptance decisions occur inside the routine.
4. The returned repair scalar is the **last pretranslation separation**, not
   the final separation or the initial global minimum distance.

The threshold is strict. The 0.7 coefficient is recovered empirical behavior,
not a derived chemical equilibrium bond distance. The carbon SI's criterion
1.7 gives a target separation 1.19 Angstrom; it is not a universal value for
mixed compositions, crystals, chemistry or physical sampling. Both modes copy
input in the independent implementation; the original true mode writes in place.

Counterexample distinguishing step 2: collinear x coordinates
`[0, .1, .2, .3, .4, 10, 12]`, criterion 1.7. The native detector returns 9.6,
while the closest arbitrary two components are 2 apart. The repair returns
10.41 and moves the last two atoms to 1.59 and 2.78. This corrected the first
agent reference, which incorrectly used all component pairs.

## Verification and boundaries

`research/ga_ssw/evidence/native-vapor-reference/comparison.json` stores
27 explicit input geometries, each run in both modes: 54 original-instruction
comparisons, including random multi-component inputs and the actual C60 MACE
last biased work and landing. Maximum coordinate and scalar discrepancies were
both 1.78e-15 in these cases. A separate execution of
`verify_native_vapor_reference.py` reproduced all 54 comparisons; results are
in `reverified.json`. Inputs and the implementation snapshot are retained.

The native harness executes the geometry function, while stubbing allocators
and checked-size arithmetic. It does not execute the LASP main program or
establish an entire native trajectory. Early allocator/import failures remain
in the research record. SIMD/NumPy reduction tie behavior, invalid/degenerate
coordinates, and original caller refresh/termination branches have distinct
boundaries; finite matching examples do not prove unrestricted release parity.
The research API currently rejects N<2 rather than inventing empty-system semantics.

## Real-system consequence

The correct counterfactual uses the completed MACE LS record's `last_atoms`,
verified identical to `quenches[-2].result.atoms`, then applies native geometry
before the true quench. The independent geometry also matches that exact input.
The resulting MACE quench/fresh costs 79+1=80 E/F; final energy is
-508.82286162350294 eV with fmax 0.00600131 eV/Angstrom. This is 3.73358 eV
below the old C58+C2 landing, but 4.95161 eV above the original search minimum.
Using the unchanged 1.6399999618530273 Angstrom graph criterion gives one
60-atom component, with 58 degree-3 atoms, one degree-2 atom and one degree-4 atom.
It is not a recovered all-three-coordinate cage or a low-energy discovery.

Evidence: `c60-native-vapor-mace-counterfactual-v2/`. A prior mistakenly selected
GFN2 intermediate cost another 45 E/F; it is retained and explicitly excluded
from this matched comparison. Total new counterfactual effort is 125 E/F.
No timing or general performance advantage follows from these different terminal
states. A terminal counterfactual is not a complete SSW run, and the original
in-quench detector policy has not been transplanted.

## Decision

Keep an explicitly optional isolated-cluster geometry component for further
SSW/LS testing. Do not promote it to a universal default, enforce connectivity
on all reactive searches, or conflate a repaired connected geometry with a
qualified cage/minimum. Prioritize the complete climbing release-state audit
and real equal-budget baseline comparisons before adding further search policies.
