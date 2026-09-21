# Periodic routing descriptor: recovered structure and explicit corrections

2026-09-10. `pamssw/standalone/periodic_descriptor.py` supplies a periodic
three-shell descriptor and three-basis similarity projection for GA routing.
It does not supply a structural identity predicate, kinetic connectivity,
new optimization policy or native descriptor parity. No PES calls were made.

## Actual Java assembly and branch dependence

Sources below are under
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/decompiled/sgn/`:

- `app_ga/GaSupport.java:145–165`: selects slow `Neighbour.getConInfo` or quick
  `Neighbour.quickGetConInfo`, then calls `Classify.sims` against configured
  bases. The quick call hardcodes neighbor multiplier 1.5; slow uses NeiSize.
- `nna/Classify.java:15–38`: computes one six-component weighted similarity
  per supplied base record, storing these as ModelSimS.sims.
- `configure/ConfigureRead.java:202–209`: loads `input/coninfo.base` and checks
  atom-row count against configured composition (except TYPE4).
- `nna/NeighbourPeriodic.java`: quick periodic image-labeled first shell,
  image-shift-summing second and third shells, per-species counts, and radial
  RMS values. `getConfigureInfo` uses RMS(r−b/2) per neighbor species; `getBaseDI`
  likewise uses RMS(r−b/2), pooled over each higher shell.
- `nna/PeriodicStructureBond.java`: quick branch accepts 0.1 < r < NeiSize*b
  after finite expansion/grid discovery. `BasicInfo` obtains b from radius sums;
  Python requires that positive bond-reference table explicitly.

The projection dimension is **the number of base records**, not intrinsically
three or four. The reconstructed `population.partition` consumes only the first
three. Python therefore requires exactly three fixed bases for this routing
interface; it does not silently fabricate or optimize bases. The caller may
provide reference structures with explicit provenance and hold them fixed for
the campaign. These are an independent basis choice unless they reproduce the
native `coninfo.base` construction. Native virtual bases can have fabricated
coordination rows and are not automatically equivalent to physical references.

## Why the slow and quick native branches are not a clean general descriptor

The supplied TYPE1-AlOH `configure.non` sets `IfQuickBond=false`, `NeiSize=1.5`,
and weights `(0.3,0.2,0.2,0.1,0.1,0.1)`. These are that example's settings,
not universal defaults. Its slow periodic branch calls
`Bond.getAllBondForPeriodic` (lines 72–115): it iterates only i<j within the
original cell, selects one minimum distance among 3×3×3 images, initializes
that minimum at 10 Å, and discards image labels. This excludes self-image bonds,
collapses multiple images to a single neighbor, and makes higher graph shells
operate on original atom identities/coordinates. A primitive one-atom FCC cell
therefore cannot have its physical twelve first neighbors in this branch.

The quick branch retains image labels, but `get23NeiCooInfo` fetches shared
AtoCooP objects from one map, writes their distance for each center, and stores
those same objects in multiple lists. Both second- and third-shell lists are
constructed before radial aggregation. Later writes can overwrite distances
needed by earlier centers/layers. This is source evidence of aliasing, not an
original-JAR quantified error fixture. It prevents treating that implementation
as a reliable definition of center-specific shell geometry.

In this sgn.jar source, AllConInfo includes n1/n2/n3 counts AND d1/d2/d3
in lexicographic sorting (the central species is absent). Thus it must not be
confused with the existing cluster Python count-only sort. Distance aliasing
and lost image identities remain separate problems even with radial tie-breaks.

## Implemented independent geometry contract

`periodic_descriptor(atoms, bond_lengths, neighbor_range)` requires nonempty,
unconstrained, fully periodic geometry with positive cell determinant, an
explicit complete positive species-pair reference-length table in Å, and a
positive dimensionless neighbor multiplier. It makes six arrays with the same
layout needed by the existing similarity calculation:

- n1/n2/n3: per-center, per-neighbor-species graph-shell counts;
- d1: per-center, per-species RMS(r−b/2) in the first shell;
- d2/d3: per-center pooled RMS(r−b/2) in the second/third shell.

A graph node is `(atom index, integer image shift)`. First neighbors use the
strict cutoff r < neighbor_range*b; second/third shells compose image shifts
and subtract all preceding shells and the zero-shift center. ASE enumerates all
required periodic images, including nonzero self images. Distances are computed
separately for every center/node and are never stored in shared mutable neighbor
objects. Exact coincident sites are rejected rather than hidden by the native
0.1 Å filter. No guessed bond radii or grid sizes are introduced.

Rows sort by central species, shell counts, then radial values. Including
central species intentionally differs from native sorting; radial tie-breaks
are present in the inspected sgn AllConInfo source. In exact arithmetic this makes the representation invariant
to atom permutation, global translation/rotation, individual periodic-image
relabeling and a unimodular cell-basis change. Floating-point near-ties or cutoff
crossings remain numerical sensitivities; the descriptor is discontinuous when
bond connectivity changes and is not an optimization gradient.

Repeating a cell preserves local row values but changes row count. Projection
is deliberately restricted to fixed composition and atom count; do not compare
a primitive/supercell descriptor by silently truncating rows. Independent
archive matching can recognize equivalent supercells separately.

`periodic_projection(descriptor, bases, weights)` returns three similarities,
using the recovered six-component similarity function. Bases must match the
same composition, neighbor parameters and corrected descriptor contract.
Weights are explicitly supplied, nonnegative and normalized by the existing
similarity routine. All six arrays are checked for matching shape, finite and
nonnegative entries. Equal or poorly separated bases can collapse routing;
the caller must inspect representation quality rather than infer useful PES
regions from nonempty partition output. No learned metric or heuristic reward
has been added.

## Verification and remaining boundaries

Four zero-PES tests exercise primitive FCC self-image coordination (12), repeated
local environments, real saved AlOH geometry under rotation/translation/image
relabeling/permutation, unimodular basis changes, and projection/composition
validation. These are mathematical/geometry checks, not a cross-system search
benchmark or original-JAR descriptor fixture.

A periodic GA controller can now set rows to `{energy, sims}` from this explicit
projection and call the existing partition implementation. It must retain basis
and table provenance and use a separate species-aware periodic structural
predicate for archive identity. Pymatgen StructureMatcher with explicit
scale/primitive/supercell/tolerance settings is one already exercised option;
its identity approximation does not turn routing similarities into barriers,
physical funnels, or proof of distinct stable minima. Native descriptor defects
are not reproduced as production modes; corrected behavior must be labeled
independent rather than exact release parity.
