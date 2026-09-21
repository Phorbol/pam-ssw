# Initial populations: source inventory and implemented seed expansion

Primary evidence: uploaded sgn.jar, CFR 0.152,
`app_ga/GaInitialStructure.java:generate0..generate4`.
An initializer supplies trial structures, not force-certified minima. Public
routing reference structures remain caller inputs, never silently replaced
by a freshly generated or failed structure.

## Implemented TYPE2/3/4 interfaces

`pamssw/standalone/initializers.py` returns InitialPopulation with structures,
per-structure origins, status/ledger, auxiliary_evaluations and physical_requests.
All origins explicitly carry certified=False; the subsequent real walker
owns initialization/quench and physical certification. No calculator or SSW
call occurs inside the initializer.

TYPE2 (`initialize_type2`) retains all supplied molecular-crystal seeds and
adds count serial ReComMC candidates built from seed zero. This is the actual
generate2 structure, with the documented deterministic correction to the
shared-state parallel native implementation. Complete disjoint molecule groups
and optional explicit periodic image lifts are required. It does not invent
molecular connectivity or a random density. The generated cell follows source
extents+.5 A, so physical relaxation/filtering remains necessary.

TYPE3 (`initialize_type3`) takes supplied nonperiodic molecular aggregates,
whole-seed energy metadata and topology. For count C it generates C CrossMC,
C rotation/recombination, C monomer reconstructions and C single-monomer
rotations; retains raw seeds; applies explicit source BLLimit; shuffles by
Fisher-Yates; and truncates to C, matching generate3 ordering. The source
attaches a bookkeeping extents+50 A vacuum cell; independent ASE retains
nonperiodic physics and does not mislabel that box a crystal. Atomically
mutable groups use the separately documented corrected complete-library
mechanism. Native Compete still requires >2 parents and nonzero energy span:
energies must not be fabricated to bypass that restriction. Its optional
`LJBase.monoOpt` branch now uses the source intermolecular LJ objective with
corrected analytic rigid-unit gradients and separately counted auxiliary work;
see initializer-final-source-closure.md. It requires explicit pair sigmas.

TYPE4 (`initialize_type4`) retains seeds and adds **one** native-count batch
of supported-cluster operators. It does not repeat until reaching C. All
fixed-support and 2D periodic corrections are documented in type4-surface-ga.md.
Generated candidates undergo an explicit full-periodic collision filter,
which improves on the unfiltered original initializer and is not claimed as
parity. Raw supplied seeds remain distinguishable from generated candidates.
Synthetic cubic LJ refinement cost is separately retained; ASE physical
requests are zero. A supplied support, adsorption composition, lateral site,
atomic radius table and cutoff table are required.

Statuses concern generation only. completed does not mean the structures
are stable, the target population survived quenching, or a valid global
search has started. Generated candidates, filtering and source indices stay
in the ledger; TYPE3 filtered/shuffled/truncated indices are explicit.

## Geometry verification on actual uploaded inputs

`python -m pytest tests/standalone/test_initializers.py tests/standalone/test_ga_operators.py -q`
passed 22 tests in 4.80 s, zero physical PES calls. Initializer-specific checks:

- XXXII172: one seed plus one molecular reconstruction, all four complete
  molecules retain their internal distances; neither is certified.
- (H2O)15: three seeds plus four operator candidates, filter/shuffle/cap to
  one output, all candidate origins retained; LJmonoOpt requires explicit sigmas.
- TiO2@Au24O4, 514 atoms: three seeds plus seven candidates from a single
  C=4 batch. The 486-atom support/cell is unchanged. A deliberately reduced
  three-evaluation auxiliary cap tests cost accounting: 90 synthetic calls
  (3 cubic families x10 starts x3 calls), not physical validation.

The geometry tests use supplied source inputs and explicit flag/limit choices.
They are not a benchmark of initialization diversity or post-quench yield.

## Algorithm inventory and remaining differences

| TYPE | Native initial input and generation | Current independent boundary |
|---|---|---|
| 0 | composition plus Unlimited, TripleTangency, SimpleCubic, IrregularBall/Ori, IrregularCage, RegularRing, RegularCage, CustomStructure | all source geometric families implemented; mixture/templates explicit, post-quench efficacy unvalidated |
| 1 | CustomStructure/ARC seed structures, repeated forced ForDoping periodic mutation, distance filtering | forced-Doping expansion implemented with terminal-batch omission corrected |
| 2 | addition molecular-crystal seeds, parent0 ReComMC | source-backed expansion implemented |
| 3 | addition aggregate seeds, CrossMC/mutations/filter/shuffle, optional LJmonoOpt | expansion, mutable libraries and corrected optional rigid-unit LJmonoOpt implemented |
| 4 | addition supported-cluster seeds, CrossLoaded/MutateLoaded | complete one-batch expansion implemented, with documented support/periodic/auxiliary-gradient corrections |

TYPE1 code requires CustomStructureMultiple=0, then invokes CustomStructure(0),
which is an input-reading convention; this must not be mistaken for a density-
free crystal generator. TYPE2/3/4 also explicitly read addition files. Only
TYPE0 supplies genuinely composition-driven generators in this source. A
claim that every variant autonomously invents an entire physical domain from
chemical formula would exceed the original algorithm as well as our port.

Remaining checks should target supplied template provenance, explicit family
allocation and descriptor/identity semantics; all geometric packing primitives
are now available in packing.py and initialize_type0_regular. They should not
be replaced by unmotivated random kicks or generic boxes under old names.

Update: TYPE1 initializer and TYPE3 optional LJmonoOpt are now implemented
with source-defect corrections; RegularRing/RegularCage/TripleTangency are
exposed as an explicit TYPE0 subset. See `initializer-final-source-closure.md`
for current formulas, boundaries and validation. The inventory rows above reflect this implementation update.
