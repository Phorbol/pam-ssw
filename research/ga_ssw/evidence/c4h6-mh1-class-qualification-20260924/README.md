# C4H6 connected-class representative qualification

CPU1473239 completed on dpn01 in4m58s, source3f1e7f9. Exactly42 fresh E/F
pairs and42 analytic Hessians; no relaxation or additional search. Frame-loop
wall time280.78s excludes imports/model startup; Slurm elapsed includes them.
All42 fresh checks meet the unchanged0.03eV/Å force and1e-6eV energy-agreement
criteria. The42 selections are one minimum-energy saved representative per
arm/seed/connected graph class in the frozen200000-request prefix, including
initial connectivity. They are not42 independent systems or42 distinct global
classes (the global connected union contains11 graph classes).

| Method | Seed61 observed connected classes | Positive internal spectrum | Seed67 observed connected classes | Positive internal spectrum |
|---|---:|---:|---:|---:|
| SSW |5|5|5|5|
| Paper LS |6|5|8|8|
| Native-inspired LS |10|10|8|7|

Forty representatives have strictly positive24-dimensional internal spectra;
the smallest positive eigenvalue is0.05034eV/Å². The two negative observations
are both graph class11: paper-LS seed61 minimum132 (−0.001058eV/Å²,
fmax0.02104eV/Å) and native-LS seed67 minimum249 (−0.014395eV/Å²,
fmax0.02301eV/Å). Their residual forces are nonzero; these signs at the saved
geometries neither establish transition states nor exclude nearby minima.
No tolerance was tuned to recategorize them. No other representative was
substituted after seeing its Hessian. Full eigenvalues and raw Hessians remain
in each frame directory; maximum antisymmetry2.85e−14eV/Å² and maximum fresh
energy difference9.10e−13eV are implementation checks, not MLIP accuracy claims.

## Scientific decision

Within this MH-1/omol model, initial structure, parameter set and two seeds,
native-inspired LS increases the number of connected graph classes having a
force-qualified, positive-internal-curvature representative at the same search
request budget (10vs5 and7vs5). Paper LS is equal in one seed and higher in the
other (5vs5 and8vs5). This supports retaining native-inspired LS as a useful
optional reaction-space proposal mechanism in this task. It does not establish
a universal winner, an optimal softening target, PBE reaction accuracy, bond
orders/radical electronic states, or rigorous stationary minima. Cross-system
C60 evidence remains mixed and its cage acceptance remains unmet.

End this C4H6 development series without parameter sweeps or additional searches.
Keep the fixed graph-observation counts in the coverage report unchanged; the
representative checks are an additional evidence layer, not a rewritten metric
or proof about every landing. Do not count fragmented-product novelty toward
connected-class gains. The pilot's force-qualified104-degree negative-curvature
butadiene remains a separate warning against equating torsion range with stable
conformer coverage; it is not a denominator from this42-frame qualification.

[Selection protocol](plan.md), [fixed manifest](manifest.json), [results](results.json),
[runner](run.py), `slurm-1473239.out/.err`; unchanged geometries and Hessians
remain on shared storage in the42 frame subdirectories. No core algorithm was
modified by this qualification.
