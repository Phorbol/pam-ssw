# Fe7C3 displacement normalization: no search gain in the bounded pilot

2026-09-11. Job1269321 completed on one V100 (4v100n01),234s,exit0.
Four arms each used1999 requests:1997 search +2 independent fresh checks.
Total7996 EFS; registered Fe cumulative69084 (=61088+7996).

The only strategy change is `cell_step_metric`: the paper lattice Frobenius
rule versus an independent relative-deformation RMS rule with the same cubic
cell scale. Both use fraction.15, five cell cycles, six direction requests,
25 partial atom iterations, Safe-total/history10, and unchanged force/stress
certificates. Inputs, source snapshot and per-arm plans are preserved.

| Metric | Seed | New certified endpoint ΔE (eV) | MC accepted | Completed atomic Gaussians | Combined step |
| --- | ---: | ---: | --- | ---: | --- |
| deformation_rms | 7 | 13.873297 | no | 10 | budget exhausted |
| deformation_rms | 101 | 11.219365 | no | 8 | budget exhausted |
| lattice_frobenius | 7 | 15.114950 | no | 10 | budget exhausted |
| lattice_frobenius | 101 | 18.803186 | no | 8 | budget exhausted |

All four candidates pass fresh force/stress certificates and differ from their
initial structures and each other under all three explicitly recorded pymatgen
tolerance profiles. This does not prove full Hessian stability or new phases.
All are above the initial energy and rejected by MC. No arm improved its best
energy or completed the combined cell+atomic proposal within its budget.

The normalized rule does not consistently reduce the high-energy preparation.
At combined-step atomic entry, ΔE changes32.15→59.77eV for seed7 and
529.17→223.67eV for seed101; volumes change646.49→564.16A³ and
371.90→451.29A³, respectively. These boundaries are matched to exact request
rows and cells by the existing zero-PES cell-energy audit. They are not final
quenched structures. Later directions can diverge after the first altered
displacement, so this is a complete algorithm comparison rather than a replay
of an identical sequence of directions.

Decision: do not promote or tune the new metric on these seeds. Retain its
explicit experimental option and frozen pilot solely for ablation/reproducibility;
the default remains lattice_frobenius. No new guard/history/threshold is added.
Return to the stopping/reference-energy lifecycle and CBD parity questions.
The four-point Hessian and this two-seed pilot do not justify a general claim
that either mode solving or step normalization can never improve SSW.

Evidence: `metric-comparison-summary.json`, each metric’s `cell-energy-diagnosis.json`,
per-arm `result.json`, `evaluations.jsonl`, fresh certificates, `source-manifest.json`,
Slurm logs, frozen source, and the prospective README. The contemporaneous
baseline completed10/8 Gaussians; do not silently replace the historical9/8
counts or claim bitwise identity of independent GPU trajectories.
