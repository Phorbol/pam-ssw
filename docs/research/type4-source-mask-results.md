# TYPE4 TiO2@Au24O4: separate direction support, first full MACE steps

2026-09-11. This514atom uploaded case now completes a full independent
fixed-cell SSW attempt under both declared masks. It is a constrained MACE
model result, not GA performance, variable-cell sampling, or DFT validation.

Source: original `TYPE4-TiO2@Au24O4/addition/add.arc` and `lasp.in`, archived in
`research/ga_ssw/evidence/type4-source-direction-mace-v100`. Atoms1..297 are
physically fixed;298..514 can relax. The second arm also excludes1..351 from
the direction problem, leaving163 atoms/489 components for soft-mode search.
Physical biased and true quenching both retain217 atoms/651 components.

Matched settings: raw input, seed29, one attempt, same MACE-OMAT-0-small SHA,
float64, maxopt500, history500, T100K, fmax0.05 eV/Angstrom. Last four values
come from supplied input. Width0.6, rotation bias100 and the remaining solver
settings are explicit independent implementation choices in plan.json, not a
claim of native parameter parity. Each arm was limited to2000 E/F including
two fresh evaluations and720s search time.

| Result | Physical mask only | Source direction mask |
|---|---:|---:|
| Completed biased stages before lower-energy exit |1|1|
| Search E/F |264|268|
| Independent fresh E/F |2|2|
| Initial energy (eV) |-4343.44551210513|-4343.44551210513|
| Landing energy (eV) |-4345.80138108836|-4345.79908749376|
| Landing minus initial (eV) |-2.35586898322|-2.35357538863|
| Fresh active fmax (eV/Angstrom) |0.0459623|0.0407259|
| Maximum mobile-atom displacement (Angstrom) |2.48749|2.53654|
| Fixed atoms/cell exact |yes|yes|
| MC accepted |yes|yes|

Both runs complete the full lifecycle and reproduce endpoint energies to
9.1e-13eV or better with a new calculator. Raw forces on the fixed substrate
remain large (~3.1eV/Angstrom), so these are deliberately constrained
certificates. They do not establish full unconstrained stationarity.
The ~0.0023eV difference between the two landings is not evidence of distinct
basins or mask superiority at the original force tolerance.

Job1249131 used one Tesla V100-SXM2-32GB on4v100n10, accountsjtu-caoxiaoming,
QOSrush-1o2gpu, allocated8CPUs/22GiB. sacct reports COMPLETED0:0 in56s;
measured per-arm wall times including fresh checks were23.39s/22.61s. These
timings are not a controlled CPU/GPU speedup measurement. Frozen source,
manifests, exact requests, in-process runtime metadata and scheduler outputs
are retained. No original LASP binary is used in either search.

Total this comparison536E/F. Earlier TYPE4 precheck/initial/fresh costs35E/F
remain separate, giving571 before additional qualification.

## Independent qualification and ordinary-descent control

Job1249278 completed all three endpoints:8283 E/F,627.91s script time.
Strict fmax0.001 quenches and fresh checks precede complete651-dimensional
active Cartesian Hessians at1e-4 and5e-5 Angstrom; no zero modes are removed
because the substrate is fixed.

| Endpoint | Refined energy / eV | Active fmax / eV/Angstrom | Smallest eigenvalue at smaller step / eV/Angstrom² | Negative count at both steps |
|---|---:|---:|---:|---:|
| Common initial |-4343.48014096464|0.00090778|-0.0106648605|1|
| Physical-mask landing |-4345.98444188059|0.00092617|0.0260771063|0|
| Source-mask landing |-4345.99257047549|0.00097777|0.0257330389|0|

Step-to-step Hessian spectral differences are about1.8e-6eV/Angstrom²,
far below the initial negative curvature. Thus the original initial force
certificate is **not a local-stability certificate**, even at fmax0.001.
The unstable mode has99.49% weight on the adsorbate. This changes the
interpretation of the raw2.35eV descent: it is not yet a qualified
minimum-to-minimum escape from that initial point.

Job1249501 subsequently completed five unbiased directional E/F probes and
three ordinary true quenches (unperturbed and ±0.05Angstrom along the
unit negative mode), all at fmax1e-5. All converge to
-4343.482540339eV within6.3e-10eV and active fmax<=8.66e-6.
Element-preserving periodic assignment gives mobile RMS differences below
9.7e-6Angstrom, supporting the same numerical endpoint. Total612 E/F,
146.43s script time. No Gaussian, search bias or algorithm parameter tuning
is involved. The ordinary descent explains0.0370282eV relative to the raw
initial minimum; the refined SSW landings remain2.50190/2.51003eV lower.
This control narrows the interpretation but does not alone establish causality
or search efficiency.

The two refined SSW landings differ by mobile RMS0.04381Angstrom, predominantly
in the mobile support (RMS0.04638 versus adsorbate0.01890). Their0.00813eV
energy separation is not evidence that the direction mask is generally better.
Both have positive full finite-cell constrained Hessians. No barrier,
long-wavelength phonon, DFT, physical ground-state or model-domain claim follows.
See `type4-direction-strict-qualification/structure-comparison.json` and
`audit_type4_endpoint_identity.py` for the element-preserving MIC assignment;
fixed support registers both structures, with no free rigid alignment.

## Repeat from a curvature-qualified initial

Job1249727 completed successfully. The tighter initial has active fmax
7.91789e-6eV/Angstrom and minimum Hessian eigenvalues0.0074465037 and
0.0074465065eV/Angstrom². Both full651-dimensional Hessians are positive;
qualification costs2607E/F. The conditional gate therefore permits the
matched repeat at the explicit research fmax0.001, with seed29 unchanged.

| Repeat | Total E/F including2 fresh | Landing energy / eV | Delta from qualified initial / eV | Fresh active fmax / eV/Angstrom |
|---|---:|---:|---:|---:|
| Physical mask only |523|-4345.97928951546|-2.49674917647|0.000766216|
| Source direction mask |413|-4345.97943897910|-2.49689864011|0.000989910|

Both complete one biased stage, true quench and acceptance; fixed atoms and
cell remain exact. Fresh energy errors are<=2.73e-12eV. This establishes that
the observed lower-energy candidates persist when starting from a point
qualified by force and full finite-cell curvature. The new landing geometries
are distinct records: positive Hessians of earlier landings cannot simply be
reused as their certificates. The new baseline landing subsequently passed both complete Hessians at its
actual saved geometry: minimum eigenvalues0.0304879023/0.0304879040, zero negative
values,2607 additional E/F. Thus the ordinary single-step run now connects
two separately force/curvature-qualified finite-cell constrained points in
this model; no transition barrier or physical dynamics is established.
The110-request difference is one paired seed, not an efficiency ranking.

The listed series cost through this repeat is13009E/F, including8283 prior
strict checks,612 unbiased controls and2607 initial-curvature rechecks.
`research/ga_ssw/summarize_type4_costs.py` rebuilds the cost ledger from terminal
artifacts; pending work is not silently counted as completed.

## Predeclared next test

Job1249979 (`evidence/type4-multistep-heldout`) first checks the new baseline
landing with both complete Hessians (3300E/F ceiling), then runs seeds7 and101,
three consecutive attempts each, with both masks from the same qualified
initial. Each of four arms has3000 total E/F including a reserve of four fresh
endpoint requests and600s; the total search ceiling is12000E/F. All parameters
are frozen to the stricter repeat, with no cap extension or post-outcome tuning.
The35min job uses one V100. Its baseline-landing qualification is complete;
all four held-out arms have now terminated (COMPLETED0:0 scheduler status,
16min26s). This checks continuous-state behavior and transfer
beyond seed29; two held-out seeds still constitute a bounded pilot. Failed
attempts, rejected candidates and validation cost remain in the denominator.

## Contact geometry of the certified-start repeat

Raw element-resolved MIC distances show a specific adsorbate change: oxygen
index500 initially has nearest support-O1.48509Angstrom, support-Ti1.94394
and Au3.04205. In the baseline landing these are2.82722,1.72349 and2.05887;
the source-mask landing gives2.82676,1.72147 and2.06544. The closest Au–Au
neighbor distance for every Au remains within2.62–2.88Angstrom in the baseline
landing. This supports an interfacial oxygen contact rearrangement in the model
rather than explaining the energy change solely by distant gold evaporation.
It is not an oxidation-state assignment, elementary reaction, or TS/pathway
claim. The exact contacts, all four oxygen environments and source atom
partition are in `type4-certified-start-control/geometry-contacts.json`, rebuilt
by `audit_type4_local_geometry.py` with zero new PES evaluations. No empirical
bond cutoff is used in this contact report.

The Au-only minimum spanning tree has longest edges2.81024Angstrom initially
and2.88703/2.88789 at the two repeated landings. This gives a threshold-free
measure of cluster connectivity to accompany the nearest-neighbor summaries.

## Direction curvature is not physical stability

The saved dimer curvature around-88eV/Angstrom² is the eigenvalue of
H_eff=H-beta*a*a^T with beta100, not an eigenvalue of the bare PES Hessian.
For seed29 baseline, n^T H n=10.34589 while n^T H_eff n=-88.41551; the
full-matrix biased residual0.01433 agrees with the recorded0.01431. The
computed direction has overlap0.999999992 with the exact lowest eigenvector
of the same independently assembled operator. This zero-PES check uses the
already charged full Hessian. Both seed29 modes retain about0.993–0.994 overlap
with their random anchors, consistent with intentionally constrained rotation,
not evidence of physical instability or a failed soft-mode solver.

Shang and Liu2013, DOI10.1021/ct301010b, Eqs5–6 and the accompanying biased
rotation discussion explicitly motivate retaining the initial random direction.
The code's beta convention and value are independent explicit settings; the
paper does not establish beta100 as generally optimal. The remaining parameter
question is a search tradeoff, not something resolved by making the rotation
residual smaller. See `audit_type4_biased_mode.py` and
`type4-certified-start-control/biased-mode-audit.json`.

## Held-out multi-step pilot: all four arms finished

All12 requested attempts remain in the denominator:9 force-certified landings
and3 budget-censored attempts. Each arm begins from the same separately
curvature-qualified initial. All13 fresh endpoint evaluations pass fmax0.001,
fixed-atom/cell invariants, and energy agreement<=1.82e-12eV. Total10167E/F
includes10154 search requests and13 fresh requests. The four offline state
audits pass exact selected-state, landing-provenance and request reconciliation;
the audit adds no PES evidence.

| Seed | Mask | Valid landings / requested | Total E/F incl fresh | Best delta / eV | Cumulative search E/F when that best was first seen |
|---|---|---:|---:|---:|---:|
|7|Physical only|3/3|1170|-3.51414394|1166|
|7|Source direction|2/3|2999|-3.56329052|1058|
|101|Physical only|2/3|2999|-4.65718835|1076|
|101|Source direction|2/3|2999|-4.77841739|840|

The direction mask finds a lower recorded best in each held-out seed, and those
bests appear earlier than the corresponding baseline's own best. That is a
bounded observation, not a predeclared target-hitting success rate or a universal
efficiency ranking. The complete cost still includes later failed climbing.
The baseline seed7 finishes three attempts early, so equal caps do not imply
equal spent cost. Energy-versus-cumulative-request envelopes, rather than only
final energies, are retained in `type4-multistep-heldout/comparison.json`.

All three censored third attempts are preserved. For example, seed7/source
completed seven biased quenches before the eighth was cut off by the global
budget; its last force max0.51956eV/Angstrom does not meet either a loose
per-atom0.005 threshold or the declared aggregate criterion. The failure cannot
be repaired by relabeling the final point converged or by ignoring its cost.

The best seed101 structure from each arm completed independent full
651-dimensional two-step Hessian qualification in job1250440 (COMPLETED0:0,
6min37s). Both are positive at both steps: baseline minimum0.0242580514 and
source-mask minimum0.0157565779eV/Angstrom² at the smaller step. The measured
energies are unchanged. Total5214E/F,391.62s script time. This qualifies the selected representatives,
not all nine new landings. No extra SSW attempts, cap extension or mask retuning
is added. The total listed TYPE4 series before that final qualification is
25783E/F, and30997E/F after the completed best-point qualification. The mask remains optional; continuous whole-search and cross-material
evidence take priority over additional tuning of this one system.
