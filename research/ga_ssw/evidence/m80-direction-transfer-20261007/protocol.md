# M80 Morse direction-transfer panel

**Status:** preregistered bounded development experiment. No search has been run by `--preflight`.

## Question and decision boundary

Does the operational difference between rotation-only recovery and recovered full direction with `c1_radius_policy="per_atom"`, previously exercised on soft Lennard-Jones clusters, remain visible in a short-range Morse cluster under the same SSW controls? The decision is limited to whether the behavior and cost are transferable enough to justify a later, separately designed study. This is not a reproduction of the published M80 global-minimum rate and cannot establish a general direction policy.

The comparison is paired by saved input. Each input is searched once with each arm. The same explicit search RNG seed is used in both arms of a pair, while the two input pairs use different seeds. Coordinates are never regenerated or repaired. The treatment difference is only the direction interface: `recovered_rotation` versus `recovered_direction` with the per-atom C1 radius policy. The shared `SSWConfig`, height, MC, local optimization, and backend controls come unchanged from `ssw-native-lj-panel-20261007/run_panel.py::settings()`; the direction setting is copied with only `c1_radius_policy="per_atom"` for the full-direction arm.

## Inputs and reference

Inputs are the saved force-qualified endpoints `final.extxyz` at:

- `../cluster-paper-reproduction-20260925/morse-compact-initials/m80-25092501/final.extxyz` (80 atoms; 1 component at `1.3 r0`; prior energy `-271.529285 eV`, `fmax=0.0082898 eV/Å`).
- `../cluster-paper-reproduction-20260925/morse-compact-initials/m80-25092502/final.extxyz` (80 atoms; components 73+6+1 at `1.3 r0`; prior energy `-249.648815 eV`, `fmax=0.0089194 eV/Å`).

Each `final.extxyz` is the force-qualified output of the prior random-structure qualification recorded in `morse-compact-initials/summary.json`. That preparation used 1006 and 666 search requests, respectively, followed by one fresh request per saved endpoint. This is prior input provenance and cost; it is not part of the new SSW search budget. Neither saved endpoint has had an SSW search. Their coordinates, including the fragmented second input, are retained byte-for-byte; there is no resampling, fragment repair, recentering, or extra preparation. The initial quench performed by the new `run_ssw` call is included in its paid request budget.

The reference is `../cluster-paper-reproduction-20260925/references/morse-80G-rho14.extxyz`, the rho=14 reoptimization of the Cambridge 80G reference shape. Its source-coordinate rho=6 caveat and independent qualification are recorded in `morse-source-audit.md` and `morse-reference-qualification.json`. It is used only as a comparison target and once for a shared fresh backend qualification, never as a starting direction. Candidate matching uses the existing `analyze_panel.compare_geometry`: connected-graph isomorphism followed by proper-rotation RMS with its existing mapping and RMS limits.

## Potential and operational settings

The direct backend is `ase.calculators.morse.MorsePotential(epsilon=1, rho0=14, r0=2.7, rcut1=100, rcut2=101)`, `pbc=False`, no constraints, and no confinement. ASE expresses the cutoffs in reduced `r0` units, so these correspond to 270 and 272.7 Å. At separations beyond the outer cutoff the pair terms underflow to zero in float64. This is the audited rho=14 potential, not the default ASE Morse truncation and not the rho=6 Cambridge input potential.

The paper-reported M80 values are `rho0=14`, `epsilon=1`, `r0=2.7`, `T=0.8`, `ds=0.6`, and maximum Gaussian count 14. The operational controls below are inherited unchanged from the current paired LJ study, with the stated per-atom C1 direction setting for the full-direction arm. The operational temperature conversion to kelvin is inherited from that study; it is not asserted to be an exact reconstruction of the paper's reduced Morse temperature. Other M80-specific details (initial ensemble, force symbol/units, and height-law realization) are not fully specified by the 2013 source. Therefore no exact paper-protocol or success-rate replication claim is made.

The force threshold is `0.05 eV/Å` for this development panel, matching the inherited paired-search protocol. The source paper states a force criterion `<0.04 ε/σ`, but leaves the Morse-specific meaning of `σ` unclear. We do not equate that source criterion with `0.05 eV/Å`; the latter is an explicit operational choice for this transfer panel, so its qualified minima do not reproduce the paper's force qualification. Connectivity uses `1.3 r0 = 3.51 Å` as a geometric component diagnostic; it does not certify chemical bonding or Hessian stability. The input and search domain requires finite, nonperiodic coordinates. Coordinate span is recorded only as a diagnostic and never rejects an isolated-cluster configuration. ASE's finite neighbor cutoff is part of the backend contract; no claim is made for arbitrarily extreme configurations beyond the compact-cluster domain represented by these inputs and observed trajectories.

## Bounded run plan

| Boundary | Fixed value |
|---|---:|
| Saved inputs | 2 |
| Arms per input | rotation; full direction with per-atom C1 radius |
| Paired RNG seeds | 26100731; 26100732 (same seed across the two arms for each input) |
| Outer attempts per arm | 100 maximum |
| Paid surface requests per arm | 80,000 maximum, including initial quench and failed work |
| Wall time per arm | 600 s maximum |
| Search requests across all arms | 320,000 maximum |
| Fresh requests | up to 2 per arm (cold initial and best) plus 1 shared reference; 9 maximum |
| Slurm layout | slots 0–3, at most 2 concurrent array tasks; 1 CPU per slot |
| Repeats, continuation, added budgets | none |

Each array slot runs exactly one arm with a fresh calculator, independent search state, monotonic deadline, and output directory. Slot 0 performs the one shared fresh reference check; slots 1–3 use the independently qualified saved reference and do not repeat that PES request. The request counter charges each attempted E/F request, including failed requests and requests served from an ASE cache; actual backend calls are recorded separately. Callback output stores initial and outer start/landing/current structures, status, paid cost, true landing force/convergence and connectivity, and candidate gates. Every new connected force-qualified minimum is retained with cumulative paid requests, so its energy-cost record supports offline best-qualified-prefix analysis. Tail records missed by a safe callback are recovered from `result.records` using the saved initial and per-record request counts. Checkpoints are written at arm end; they do not authorize restart or continuation.

The only target early stop is a new minimum meeting all of: `E <= -340.811371 + 0.001 eV`, convergence with `fmax <= 0.05 eV/Å`, one connected component at `1.3 r0`, and same-geometry classification against the reference. The returned best and initial structures are then independently evaluated on fresh Morse calculators. A candidate is called a confirmed 80G hit only if the fresh best also passes these gates. A search status such as `completed`, `paused`, or `failed` is execution state, not a scientific result.

## Interpretation and stopping

The 2013 paper reports 8/100 M80 trajectories reaching the GM within 2,000 SSW steps. A 100-outer-attempt panel is much shorter than that benchmark and uses operationally inherited controls, so no-hit outcomes are foreseeable and cannot estimate the published hit frequency. This small panel can support operational end-to-end observations on these two saved starts: paid cost, qualified connected lower-energy landings, comparative outcomes within the two pairs, and existence of the GM if an independently fresh-qualified hit occurs. It cannot support success-rate replication, an arm-wide effectiveness estimate, universal direction retention or promotion, nor a claim that any change transfers to other potentials or sizes.

All errors, failed requests, incomplete attempts, cap censoring, and wall censoring remain in the output. Compare the two arms within each saved input and show both pairs; do not pool them as independent random starts or omit the fragmented initial. After these four bounded runs, stop. Any longer or repeated panel requires a new protocol and decision based on these observations.

## Reproducibility

`run.py --preflight` is restricted to imports, input/reference shape and hash checks, settings inspection, geometry-only input diagnostics, and a dummy-calculator ledger-contract check. It must report zero real PES requests and must not construct or evaluate a real Morse calculator. `run.py --slot N --output DIR` is the explicit one-arm execution entry point; the Slurm array is `0–3%2`. Each slot snapshots the runner, this protocol, direct imported source files, checkout commit/tree provenance, import paths, effective settings, exact saved-input hash, and prior qualification summary into its run output. Existing unrelated dirty files in the checkout are not modified by this study.

## Source boundary

The paper-reported M80 target/cost figures and reproduction gaps are documented in `../cluster-paper-reproduction-20260925/morse-source-audit.md` from Shang and Liu, *J. Chem. Theory Comput.* 2013, 9, 1838–1845, DOI `10.1021/ct301010b`, §3.3 and Table 1. The Cambridge rho=14 target energy and rho=6 source-coordinate limitation are also captured in that audit. These facts motivate the bounded question; they are not evidence that this operational panel reproduces the paper's ensemble or rate.
