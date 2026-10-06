# C60 source #3: LASP external-input qualification

Question: can the newly qualified, published non-Ih C60 #3 be supplied to
the existing LASP fixed-cell external ASE interface on the same MH-1/omol
potential without a material boundary-condition or file-protocol change?
This is input/interface qualification, not a search or algorithm ranking.
The existing C60 vacuum check covered old inputs, not this new geometry.

Use the exact archived `final-candidate.extxyz` from
`../c60-ls-source3-qualification-20261007/qualification-1664282/` (SHA256
`dcc81d45c4e7193b306a0c1957b49c19e44751231fa95fef75201183280f0758`).
Do not relax, rotate, relabel, redraw, or replace it. Keep its centered
50 Angstrom diagonal storage cell. Compare pbc=False with pbc=True at the
same coordinates. Backend: cached MH-1 SHA256
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`,
omol head, float64 CUDA, cueq/oeq disabled. The independent Python search
continues to use the nonperiodic geometry.

Predefined gates: finite energy/forces, absolute energy difference <=1e-4
eV and maximum force-component difference <=1e-4 eV/Angstrom, using the
previous qualified vacuum-comparison tolerances. Report the nearest
inter-image atom separation and model cutoff; require images to lie outside
the model interaction range. A pass covers this geometry only, not every
later search configuration or equivalence of the two search algorithms.

Then supply the periodic input ARC to the unchanged LASP binary through
`lasp_external_ase.run_lasp`, existing external client, and bounded-process
supervisor. At most one successful E/F callback is allowed. Preserve the
raw external.coord, returned E/F, fixed-cell/composition checks, binary and
used-helper provenance, actual imported paths, process output and status.
Compare the returned geometry/E/F with its separately evaluated periodic
input; coordinate transmission tolerance is 1e-4 Angstrom per component,
with the same E/F tolerances above. A protocol or geometry mismatch blocks
subsequent native search.
No expiry/security bypass, cell relaxation, constraints, or public API
changes are part of this task.

Resource boundary: one V100, five minutes, process supervision <=60
seconds for LASP, two direct E/F evaluations plus at most one successful
external callback (three paid requests total). Budget-rejected callback
attempts are separate, retained in the record, and do not invoke MACE.
Record issued, paid and actual calculate counts separately, including
execution failures. No automatic retry or search extension. A dummy
Calculator/child preflight can check directory creation, ARC serialization
and the real callback path without MACE or physical PES evaluation.
The caller also enforces its three-paid-slot cap before invoking the
Calculator, including failed E/F attempts: the archived helper's successful
response limit alone does not bound failed evaluations. A repeated-failure
dummy probe checks this boundary. Native budget-stop exit and supervisor
cleanup status are reported separately from callback qualification.

Decision: passing both the vacuum and callback gates permits a separately
specified bounded native comparator; it does not itself authorize claiming
LASP or Python superiority. Failure distinguishes boundary/model mismatch
from executable/file-contract failure and is diagnosed without changing
the concurrent SSW/LS tests.
