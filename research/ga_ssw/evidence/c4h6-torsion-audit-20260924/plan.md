# C4H6 coverage-metric diagnostic

Question: does the existing element-labelled connectivity classifier collapse
trans/cis butadiene, which LS-SSW §3.1 distinguishes as A/B? If so, what torsions
were actually present in the archived six-arm GFN2 development data?

Protocol fixed before analysis: read all saved minima from all six arms of
`../c4h6-ls-reaction-coverage-20260912`, including their original convergence and
fresh-check flags. Reuse the archived analysis's graph rule (native H/C lookup
length + 0.1 Angstrom). Select only complete graphs isomorphic to ASE G2
butadiene. On their carbon four-node path report the C-C-C-C dihedral and its
cosine; positive/negative sign means geometric cis-like/trans-like, not a
stationary isomer or elementary reaction. Preserve exceptions and zero-sign
ambiguity. No newly fitted angular acceptance threshold.

Control: ASE G2 trans-butadiene, a constructed zero-dihedral geometry obtained
by rotating one central-bond component, and reversed atom indexing. Expect
the first two to share connectivity but have opposite torsion cosine signs;
reindexing must preserve cosine to numerical arithmetic precision. The
constructed geometry is not a minimum and is not a new physical test case.

Possible outcomes: (a) old graph-matched frames include both torsional regions,
so graph-only counts hid conformational exploration; (b) only one region is
present, so the classifier limitation exists but did not hide an observed
region in these saved frames. Neither outcome proves LS efficiency; this is
reuse of existing development trajectories, with correlated/repeated minima.

Budget: zero calculator calls, one CPU-MISC task up to five minutes; no xTB,
MACE, search, relaxation or geometry changes to old evidence. New output only
in this directory, refusing overwrite. Core checkout at preparation:
82da3e33e5dc73a4ff343dd30ba438b527cc4ba3. New harness and plan are archived
with resulting analysis; no changes to the graph classifier or search code.

Acceptance: control checks pass; all six archived arms accounted for; every
selected frame retains source path/index and numerical qualification. Counts
are observations, never distinct isomers or independent success rates.
