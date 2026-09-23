# Three saved C4H6 structures: curvature qualification

CPU1470561 completed the frozen three-frame screen using MH-1/omol, float64,
CPU (dpn01). Source commit91af36e. Slurm elapsed2m21s includes Python/module
startup before the script timer; the22.76s below is not full job wall time. Three fresh E/F evaluations plus three analytic
Hessian evaluations,22.76s in script (including model loads/hash). Hessian cost
is reported separately, not equated to three force calls. No geometry relaxation,
parameter change, search rerun or automatic expansion occurred.

| Saved pilot structure | CCCC torsion (degrees) | Fresh fmax (eV/Å) | Lowest internal curvature (eV/Å²) | Negative internal eigenvalues |
|---|---:|---:|---:|---:|
| SSW min0 |180.00|0.025288|+0.231112|0|
| paper LS min1 |25.23|0.014229|+0.081919|0|
| native-inspired LS min3 |103.59|0.021802|−0.191458|1|

Fresh energies match the archived GPU values in the reported float64 outputs.
Each30×30 Cartesian Hessian was projected onto the24-dimensional space
orthogonal to global translations/rotations. Maximum Hessian antisymmetry was
at most2.85e−14eV/Å², much smaller than the observed negative curvature.
Analytic Hessians took6.63–6.85s each on CPU; fresh E/F took0.24–0.79s.
No finite-difference step or numerical stability threshold was selected.

**Observed:** the approximately104-degree saved geometry meets the common
force criterion but has a negative internal direction. The other two selected
geometries have positive internal spectra. The nonzero residual force prevents
calling any frame an exact stationary point; one negative eigenvalue here does
not establish a certified first-order saddle, nor exclude a nearby minimum.

**Decision:** keep force-qualified observations and stable-isomer claims
separate. The pilot torsion range cannot by itself prove new stable conformers.
This does not establish a native-LS-specific defect: all methods use the same
force-based quench criterion, and the frames were deliberately selected rather
than randomly sampled. Do not tighten fmax, add a per-step Hessian gate, or
change the running six-arm protocol on this evidence. If its final coverage
suggests additional connected classes, qualify the actual claimed representative
structures before interpreting that as stable-isomer discovery. Counts of
observed graph classes remain observations under their original fixed definition.
The three-frame screen neither certifies nor rejects the other36 pilot frames.

Evidence: [frozen plan](plan.md), [runner](run.py), [results](results.json),
`ssw-min00/`, `paper_ls-min01/`, `native_ls-min03/` (inputs, spectra and raw
Hessians), `slurm-1470561.out/.err`. Runtime warnings concern optional pynvml
package deprecation and loading the explicitly trusted local model; all three
calculations completed. No dependency was changed.
