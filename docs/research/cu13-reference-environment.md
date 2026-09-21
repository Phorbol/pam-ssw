# Cu13 direction-only reference environment audit

This is a bounded replay audit for `tests/standalone/test_direction_only.py`.
It does not alter the reference JSON, test tolerance, or solver configuration.
The executable audit is `research/ga_ssw/audit_cu13_reference_environment.py`;
its scalar outputs are stored beside the evidence logs.

The ordinary current checkout environment used NumPy 2.5.2 and ASE 3.29.0.
Both `ritz` and `dimer` reproduced the stored final coordinates exactly. The
required no-user-site environment used NumPy 2.0.2 and ASE 3.26.0. With the
frozen source prefix and with the current source prefix, both solvers retained
the same `biased_quench_failed` status and essentially identical directions,
but differed from the stored final coordinates by 0.16285519498 Angstrom
(`ritz`) and 0.03385901866 Angstrom (`dimer`).

The frozen run imported `pamssw.standalone` from the evidence source, while the
current run imported it from this checkout. Relevant frozen/current source
files (`paper_reference.py`, `direction.py`, `cluster_frame.py`, `surface.py`,
`gaussian.py`, and `relax.py`) were byte-identical in this audit. Removing the
new package-level Ritz export from a temporary copy did not change the no-user
results, so the observed discrepancy is not attributed to that API change.

These measurements establish an environment-sensitive replay discrepancy. They
do not identify which environment produced the archived JSON, because that
metadata is absent from the reference artifact.
