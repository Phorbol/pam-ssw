# TYPE3 mutable internal-unit libraries

Source: uploaded sgn.jar, CFR decompilation of
`ga_monomer/MutateMonomer.java:monomerReconstruction`, and its calls to
`ga_cluster_cell/Cross`, `Mutate.getMutateStructureForDoping` and MonomerBase.
The implementation is independent Python and makes no calculator calls.

For each changeType=1 group, native code extracts that group's atoms from
all parents, attaches each **whole parent's energy metadata**, generates
an atomic crossover/mutation library and chooses library member i for
reconstruction i. changeType=0 groups always come from parent zero. It then
centers the fragments and docks with min distance 1.5 A and accuracy 10.
This is not a learned conformation library or molecular force-field search.
Reconstructed units may have different internal geometry/connectivity.

`mutable_monomer_library` implements the source Cross plus forced Doping
schedule even when a unit contains only one element. The Doping quota differs
from automatic pure-element TYPE0 mutation, so blindly calling mutate_type0
would be incorrect. The individual cut, exchange, disturbance and corrected
reinsertion primitives are shared with their existing source-backed ports.
Reinsertion retains the previously documented corrected collision predicate,
not the native loop that accepts close contacts. Small units are rejected
only if a nonzero quota actually requests five-atom removal.

## Cardinality defect and explicit repair

For source request n, crossover count is c=floor(2n/3), mutation request m=n-c,
and actual library size is

    L(n) = c + 3 floor(m/4) + 5 floor(m/8).

The Java code subsequently indexes n members regardless of L(n). It therefore
underflows for common small requests: L(1)=0, L(2)=1, L(10)=9, L(21)=17.
The first self-sufficient positive source request is 22, where L(22)=25.

`native_quota` explicitly rejects underflow before generating anything.
The independent production policy `complete_library` selects the smallest
integer q>=n with L(q)>=n, executes that complete source library schedule,
then takes the first n entries. Thus n=1 uses q=2 and one atomic crossover.
This deterministic integer correction adds no random fallback, energetic
score, adjustable allocation weight or tuned parameter. All generated and
discarded work remains in the library ledger. The repaired fractions can
differ from the impossible original small-n request; this is not claimed as
trajectory parity. Pair/cut/insertion attempts remain bounded explicitly.

## Topology and lineage

An atomically crossed group can contain atoms from several parents.
`group_parent_indices` therefore contains None for mixed groups, not a
fabricated single parent. Details carry exact `atom_parent_indices`, original
`source_atom_indices`, and per-group `group_parent_sets`. Within each group,
a stable species-based reordering restores the original ordered element
topology without changing physical coordinates/composition. This keeps
subsequent controller checks meaningful; the same permutation is applied to
lineage. Rigid groups remain unchanged internally. The library ledger is
shared once per mutation call/group; repeated references from different
candidates do not represent repeated computation.

`changeType=2` is not silently interpreted as an RC internal torsion mode:
the uploaded reconstruction code builds libraries only for type 1 but uses
all nonzero types as library indices. That path needs separate source
resolution before supporting it. Disjoint physical units remain distinct
from overlapping RC bodies.

## Verification

`tests/standalone/test_ga_operators.py`: 19 passed, 2.32 s, zero PES.
Actual uploaded (H2O)15 coordinates test one atomically mutable water plus
14 rigid waters: one-output repaired library succeeds, all rigid internal
distances remain invariant, ordered composition and exact lineage survive
docking. The flag change is an explicitly selected test condition; the
uploaded segmentation itself marks waters rigid. A 28-atom Au24O4 loaded
fragment from the actual TYPE4 input tests a 22-member library, 25 generated
candidates, three discarded entries and retained reinsertion work. These
are geometry/code checks, not stable molecular reconstructions or search
performance evidence.

## Remaining initializer inventory

`app_ga/GaInitialStructure.java` is the primary source. Only TYPE0 combines
composition-based Unlimited/TripleTangency/SimpleCubic/ball/cage/ring/custom
generators. TYPE1 reads custom ARC structures and mutates their cells;
TYPE2 reads addition structures and appends ReComMC; TYPE3 reads addition
structures then CrossMC/three MutateMonomer families, filters/shuffles,
optionally applies LJBase.monoOpt and truncates; TYPE4 reads addition
structures and appends CrossLoaded/MutateLoaded outputs. Consequently,
all-variant support does not imply the native algorithm can create molecular
connectivity, a crystal cell or a supported-cluster substrate from composition
alone. These must remain explicit physical inputs.
