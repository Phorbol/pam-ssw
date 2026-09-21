# Native local pair geometry: scope and falsifiable follow-up

2026-09-11. Question: does the release's cooperative and independently normalized
local direction avoid the extreme two-atom concentration observed in one C60
paper-direction trial? That causal claim is untested. The first deliverable is
only an independent geometry helper, keeping all existing walker defaults.

Recovered source: archived LASP ELF SHA256
bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704,
localatompair_mode0x6e4490, neighboringlist0x580c50, species_radius0x58f680.
Caller0x5d7dd7 normalizes the resulting local vector before coefficient mixing.
Evidence resides in research/ga_ssw/evidence/native-cluster-control-generator.
The numerical probe executes those geometry instructions with a prescribed RNG;
it neither invokes LASP main nor evaluates a PES.

For an isolated unconstrained atom array with explicitly chosen pair(i,j),
construct opposing unit vectors toward the other atom. Reject the pair when
its separation is below max(0.6*(radius_i+radius_j),0.7Angstrom). For each pair
endpoint, enumerate neighboring atoms in index order with separation strictly
below radius_center+radius_neighbor+0.5Angstrom and first Cartesian freedom-mask
entry nonzero. Perform up to neighbor-count uniform draws of slots with
replacement, clearing only successfully selected slots; stop at four successes.
A selected neighbor receives0.8 times a unit displacement toward the opposite
pair endpoint only when its distance to that opposite endpoint exceeds3Angstrom.
The marker-1 reverses displacements; the final componentwise freedom mask is
applied. Return raw geometry; normalization belongs to the caller.

Every constant above is an empirical release value, not a first-principles
parameter or a proposed universal default. The species routine has explicit
values for a subset of atomic numbers, otherwise1.25Angstrom. Preserve those
values for a source-comparison component; do not silently replace them with an
ASE radii table. Species-table extraction and original helper instructions must
be checked together, including threshold equalities and asymmetric neighbors.

No periodic extension, pair-selection policy, whole native RNG, group branch,
setconstraints projection or complete gen_randommode parity is asserted. The
helper must reject unsupported periodic input, use explicit zero-based Python
indices, and expose sampling records. It requires O(N) geometry work for two
neighbor lists, plus small bounded sampling, and zero calculator calls.

Next scientific comparison must separate local normalization from neighbor
cooperation, retain the existing ordinary SSW baseline and initial structures,
and charge full E/F cost through true quench and structural classification.
One preselected C60 seed may diagnose implementation and failure mechanism;
it cannot establish generalized benefit. Retain the helper as experimental;
remove any proposed default integration if broader matched tests show no useful
intact low-energy basin discovery per cost. Do not tune the radius or3Angstrom
threshold retrospectively against the existing C60 endpoint.
