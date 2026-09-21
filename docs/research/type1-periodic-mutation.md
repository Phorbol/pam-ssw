# Recovered TYPE1 periodic mutation slice

2026-09-10. Implemented `pamssw/standalone/periodic_ga.py:mutate_type1` as a
standalone geometry generator. No PES calls, Java execution, frozen gate edits,
crossover or GA controller changes were made.

Source files under
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/decompiled/sgn/`:

- `ga_cluster_cell/MutateForCell.java`: disturbance, species exchange, pure and
  doping batch methods, complete file reviewed.
- `ga_Interface/TYPE1.java:48–66`: mutation called for the first region and the
  remaining parent regions separately, each with minGA/2; periodic bond filtering
  follows crossover/mutation concatenation.
- `contruction/Model.java:78–89`: increasing-energy comparison.
- `other/TemTools.java:236–273`: cloning of coordinate objects and PBC cell.

## Closed semantics

The periodic Java class directly changes Cartesian coordinates while retaining
the chosen parent's lattice. It does not first convert to fractional coordinates
or wrap the mutation. This is a source-derived **periodic mutation** operation;
no cluster crossover, artificial vacuum cell or Cartesian docking has been
substituted for a crystal operator.

Disturbance chooses an atom with replacement per move and independently adds
`range*(U-.5)` to each Cartesian component. The same atom may be chosen many
times. Exchange performs 10N random pair draws and swaps atomic numbers at
unchanged sites; self-pair draws and same-element swaps are retained.

Parents are stably sorted by supplied physical energy. The Python code preserves
caller order and returns original caller parent indices. It requires the same
composition but permits different parent lattices. Offspring are pure geometry:
no inherited energy/force cache, calculator, or per-site custom arrays.
Unconstrained full-PBC structures with positive determinant are required;
explicit isotope masses are outside this species-exchange contract.

For source argument n, with integer division:

| Class | Mutation batches |
|---|---|
| Pure | n/4 best N/10 moves at range .3 Å; n/4 best N/2 moves at .3 Å; n/2 random-parent N/2 moves at .5 Å; n/2+1 random-parent N/10 moves at .5 Å |
| Doping | n/2 random-parent exchanges; n/8 best N/5 moves at .3 Å; n/8 best N/2 moves at .3 Å; n/4 random-parent N/5 moves at .5 Å; n/4 random-parent N/2 moves at .5 Å |

These are empirical release constants, not derived optimal physical settings.
The actual returned number is not n: n=8 returns 13 pure or 10 multicomponent
children. The pure n=0 branch still returns one sparse mutation, which may be a
no-op when N<10. These source quirks are preserved and explicitly tested.

## Verification and boundaries

`tests/standalone/test_periodic_ga.py` uses the existing 26-atom AlOH input,
checks caller immutability, parent-cell retention, energy sorting/lineage,
composition, exact scalar-draw Cartesian arithmetic, full species-swap draw
count and batch-count quirks. Pure cases reuse that geometry with one element
solely to exercise source branching; they are not material models or scientific
tests. No calculator is attached or called. These are source arithmetic and
contract tests, not original-JAR numerical parity fixtures. NumPy seeds do not
reproduce the Java RNG stream.

The complete TYPE1 GA remains absent: periodic crossover/pool selection,
periodic geometric filtering, first/other-region population orchestration,
periodic descriptors/archive matching and a VC-compatible controller are still
required. TYPE2 and TYPE4 are untouched. The already recovered `CrossForCell`
uses cell-aware fractional coordinate transfer for one gamete, but its entire
cut/frame behavior was not closed or ported in this bounded slice. No mutation
candidate is claimed chemically valid, force-converged or a new minimum.

## Followup: complete independent TYPE1 proposal generation

The earlier mutation-only status above is superseded for **proposal generation**:
`build_periodic_pool`, `cross_periodic_pool`, `periodic_collision_free`, and
`propose_type1` now implement the full geometry batch. A GA search controller,
periodic descriptor and structure archive remain separate missing components.
TYPE2/4 are still untouched.

Additional source reviewed: complete `CrossForCell.java`; the shared `Cut.java`
and `CutBasicAbstract.java`; `Compete.java`; `Crystal.java` Cartesian/fractional
conversion, getCM, pullBackCell and expandCell; `GA_Ab.java` periodic collision
filter and selectParent; and `TemTools.randomInt`. Both original TYPE0 and TYPE1
literally use the same Cartesian Cut class. Reusing the independently ported
cut primitive is therefore source-grounded. Periodic cell transfer and filtering
are implemented separately, not inferred from cluster docking.

The native pool uses 100 times the selected-parent count as slots, draws both
parent-index columns before cutting, then does ten cuts per slot from column
zero. Compete weights use its existing recovered formula (>2 parents with a
nonzero energy span required). Parents above 100 are selected using the best
33 plus 67 random draws with replacement; the source random-index upper bound
excludes the last energy-sorted parent and this selection quirk is retained.
Cuts center private coordinates, perform the shared random rotation and balanced
plane cut, then separate halves by ±0.3 Å. Their parent cells are retained.

Java encodes cells as lengths/angles, so Python explicitly rotates each parent
cell **and its atoms** to the corresponding canonical lower-triangular ASE row
frame at pool entry, preserving fractional positions and physical periodic
geometry. Repeated source centering is done on private copies. Native mutation
may see parent centering as an incidental shared-object side effect; Python
mutation preserves caller coordinates. That difference is a uniform translation,
not an assertion of exact coordinate-output/JVM random-stream parity.

Composition-compatible gametes are sampled with a finite caller budget.
One cell is selected with probability one half; the other half's fractional
coordinates are transferred to that cell. The selected half comes first in atom
order. The output uses Java's signed fractional remainder (`fmod`), which can
leave negative fractional coordinates, rather than silently changing to [0,1)
wrapping.

### Two intentional corrections, with no compatibility switch

`CrossForCell.joint`'s daughter branch sets the output PBC from `ova[i]` but
builds transferred coordinates in `ova[j]`'s cell. This is a real mixed-cell
metadata mismatch. Independent Python consistently uses the selected daughter
`j` cell for transfer, output and remainder. Each crossover records the selected
cell and its parent index; the source defect is not offered as a production
mode. A deterministic scalar-draw test selects i=0,j=1 with differing cells
and checks the corrected coordinate/cell result.

Native filtering checks direct pairs and a finite 2×2×2 replicated block.
Independent `periodic_collision_free` checks **all periodic neighbors within
supplied element-pair lower-distance cutoffs**, including nonzero self images.
There is no guessed radius or fitted threshold. A one-site skew-cell test has
short vector −2a+b, missed by the native finite block; the full periodic filter
correctly rejects it at the explicitly supplied threshold. Empty limits
explicitly disable the filter, as with existing proposal APIs. This is a
corrected geometric filter, not a native filter-parity claim or chemical
validity certificate.

### API and verification

```
propose_type1(parents, energies, regions, rng, *, min_ga, bond_limits,
              max_batches, max_cut_attempts, max_pair_attempts,
              slots_per_parent=100, cuts_per_slot=10)
```

`regions` is a nonoverlapping partition of caller parent indices with at least
two nonempty regions. The first supplies best-region mutations; other regions
supply the other mutation population. Crossover uses flattened selected parents.
Each batch produces minGA crossovers and mutations from both populations with
argument minGA//2, filters the whole batch, and appends all passing candidates
without truncating. Empty batches and finite sampling/batch exhaustion are
explicit return statuses, with generated/passing/filter-rejected counts when a
batch finishes. A thrown malformed input is not converted into a fallback.

`PeriodicCandidate` records atoms, operation, caller parent index per atom,
source atom index per atom, and details. Exchange mutations propagate atom
origin indices through all species swaps. Pool density overrides are available
only as explicit, recorded workload choices; the defaults match native density.
The small tests use lower density and make no native trajectory or efficacy
claim.

Eight tests now pass, including the actual saved AlOH input, a complete batch
with crossover/exchange/disturbance, parent/source-atom composition provenance,
a fixed scalar-draw mixed-cell crossover, and the skew self-image collision
case. All tests use zero PES evaluations. These are source arithmetic fixtures;
an original-JAR crossover coordinate fixture has not yet been obtained.

### Remaining controller/descriptor contract

Do not feed periodic offspring into `legacy_descriptor.cluster_descriptor`.
A periodic controller needs a declared periodic structural representation for
region partitioning, plus a separate physical duplicate predicate. An optional
Pymatgen `StructureMatcher(scale=False, primitive_cell=True,
attempt_supercell=True)` is usable for approximate species-aware structure
identity, with explicit tolerance sensitivity; it does not itself provide the
legacy DCCD coordinates or energy-region acquisition rule. The implementation
must preserve candidate composition/lineage, both accepted and rejected valid
landings, and all quench/walk costs. The choice of fixed/PQC/blocked/joint/LS
walker and force/stress certificate must be explicit. Current proposal support
alone must not be exposed as completed periodic GA-SSW search or native DCCD
parity.
