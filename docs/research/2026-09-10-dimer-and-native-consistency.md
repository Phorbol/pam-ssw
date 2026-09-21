# Dimer integration and the native energy/force consistency question

Continuation of the independent ASE SSW family research. The default remains
paper/Ritz; an explicit `rotation_solver='dimer'` now selects the independently
coded plane-minimizing dimer variation for SSW, LS-SSW and GA inner walks.
No uploaded executable is called by either production solver. Native oracle code
remains in research only.

## Paired real-system workflows

Configurations and initial geometries were saved before computation. No search
parameters were tuned after seeing the paired results. These are tiny workflow
experiments, not a performance benchmark or global-minimum campaign.

Cu13/EMT: 3 seeds, 2 outer steps each, 2 Gaussians maximum. Both solvers completed
all 3 runs and all 18 initial/landing records passed fresh-calculator 0.01 eV/A
force checks (including repeated initial records). Requests, excluding 3 fresh
checks per run:

| seed | Ritz | dimer |
|---|---:|---:|
| 3 | 293 | 304 |
| 17 | 213 | 216 |
| 20260909 | 237 | 309 |

C60/GFN2-xTB: 1 paired LS outer step, 2 Gaussians, previous force-qualified initial
structure and LS parameters reused. Ritz:137 requests,51.35s; dimer:153 requests,
54.72s. Each has two additional fresh calculator checks. Both completed and both
true endpoints met 0.01eV/A. Landing energies differed by about1.21e-9eV; no new
basin or dimer advantage is demonstrated. The larger request count cannot alone
explain system-independent efficiency; there are too few/too short trajectories.

Evidence: `research/ga_ssw/evidence/dimer-ritz-cu13/` and `dimer-ritz-c60-ls/`;
input, plan, script snapshot, per-run complete result and independent checks are
saved. One Cu13 harness launch failed before any E/F call because a serialization
import unnecessarily loaded optional tblite; the harness now serializes its own
results. No potential fallback or search-rule change was made.

## Important reproduction boundary

The whole-addgaussian instruction oracle is reported separately. Its observed
old-Gaussian force weighting must be judged against its reported energy, with
parameters frozen during derivative checks. A force field may be conservative
for a different energy while being inconsistent with the returned energy.
Therefore use 'reported energy/force mismatch', not an unsupported blanket claim
of nonconservative force. The full runtime may contain further transformations;
this function oracle does not establish their absence.

The primary implementation must preserve E/F consistency. Faithful legacy
behavior should remain separately named and testable, especially if a library
optimizer uses both reported energy and forces in a line search. Native history,
MC factor20, archive ordering and Gaussian effects must be isolated before any
claimed speed comparison with PAM. No evidence links a numerical discrepancy to
the FPACK expiration guard; see `native-expiry-review.md` for its distinct scope.

Validation before final native-oracle regression additions: 129 relevant tests
passed, including the optional original MC instruction comparisons. Existing
ASE/NumPy deprecation warnings remain. No GPU/HPC job, merge or push occurred.

Whole-function result was rerun independently by the parent agent to
`/tmp/pam-addgaussian-root-check.json`:48 cases, max energy formula error
1.7763568394002505e-15, max double-old-force formula error
1.3322676295501878e-15, max mismatch against returned-energy finite differences
0.22611608408673434. Detailed persisted inputs and all1656 FD evaluations are in
`native-addgaussian-emulated/result.json`. See
`native-addgaussian-instruction-oracle.md`; the previous missing-projection
interpretation is explicitly superseded. No compensation in the full native
runtime has yet been demonstrated or excluded.
