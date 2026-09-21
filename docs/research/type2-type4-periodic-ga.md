# TYPE2 molecular-crystal proposal and TYPE4 source boundary

Source: uploaded sgn.jar, CFR 0.152 decompilation under
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/decompiled/sgn`.
This record concerns geometry only: no energy, force, stress or global-search efficacy claim.

## TYPE2 implemented

`ga_Interface/TYPE2.java:getGA` uses floor(G/2) CrossMC followed by
G-floor(G/2) ReComMC(parent0). The same CrossMC class is used by TYPE3;
sharing that exact cut/dock primitive is therefore source-supported.
Reconstruction is distinct: TYPE2 docks with accuracy 3, TYPE3 uses 10.
CrossMC docks with minimum separation 1.5 A and accuracy 5. Both TYPE2
branches generate a **new orthogonal cell, Cartesian extents + 0.5 A**;
neither interpolates parent cells nor affinely strains individual molecules.
These numbers are empirical source constants, not derived universal parameters.

`molecular_periodic_ga.propose_type2` implements the complete proposal batch,
with explicit caller RNG and bounded sampling. `reconstruct_type2` exposes
the standalone reconstruction. Groups are complete physical molecules,
contiguous and disjoint, not overlapping RC bodies. Input coordinates must
already be a consistent molecular Cartesian lift; optional integer N x 3
image shifts specify it explicitly. No guessed connectivity/unwrap is hidden.

The native parallel ReComMC method rotates shared mutable fragments across
parallel candidates. Python implements its deterministic serial sibling:
cumulative rotations, 10M random swaps, sequential docking, then restoration
of original molecule-ID order. This is an intentional race correction.
Periodic collision filtering uses the separately documented full ASE image
geometry and supplied species cutoff table, not native finite 2x2x2 images.
All batches, rejected candidates and exhausted budgets remain visible.

Validation: `python -m pytest tests/standalone/test_molecular_periodic_ga.py -q`
passed 3 tests (4.68 s), zero PES calls. Fixture `type2_xxxii.extxyz` comes
from uploaded `TYPE2-XXXII/addition/add.arc`, first frame; segmentation is
four contiguous 43-atom molecules (172 atoms, C84H68Cl8N4O8). Tests preserve
all intramolecular pair distances and source molecule identity through both
branches, verify explicit image relabeling, and reject overlapping groups.
This qualifies geometry and interfaces, not energetic plausibility.

## TYPE4 is a separate surface problem

`TYPE4.java` standardizes a supported cluster, calls CrossLoaded(G/4), then
MutateLoaded(G/4,G/4,G/8,G/4), translates the entire structure to x/y minima
zero and z minimum 1 A, and uses a nonperiodic final distance filter.
`CrossLoaded` cuts only the loaded cluster, checks exact composition, then
reloads it onto a selected support: 2 floor(n/3) SuitLoad orientations at
accuracy 30; the remainder direct LoadedHandle.load. `SuitLoad` selects
among 6*accuracy rotations by minimizing sum(z)-N min(z), then approaches
the support in 0.125 A steps with contact cutoff 2.0+0.125 A.

Mutation branches are rotating reload, 20 percent atom disturbance with
Cartesian width 2 A plus species permutation, height-split surface
reconstruction, and four geometric rebuild families. The latter two require
separate source closure and cannot be aliases of molecular-crystal TYPE2.
LoadedHandle.load also corrupts group indices by appending loaded indices
to the support list; a Python topology adapter must correct this explicitly.

Actual uploaded example: `TYPE4-TiO2@Au24O4`, 514 atoms, segmentation support
1..486 and loaded atoms 487..514. Its lasp.in separately fixes atoms 1..297
and contains fixatommode 1..351: segmentation alone does not prove all 486
support atoms were frozen in native SSW. Any independent fully frozen
support mode must be explicitly selected and distinguished from this input.

Follow-up: TYPE4 source closure, all geometric families and complete bounded
proposal/controller are now implemented; see `type4-surface-ga.md`. Its
source-based corrections, synthetic auxiliary cost and surface routing/identity
requirements remain explicit and differ from original Java trajectory parity.
