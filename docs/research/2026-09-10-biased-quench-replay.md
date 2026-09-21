# Failed biased-quench replay: line search is only a partial explanation

## Experiment

All 31 biased-quench failures from the direction-only Cu13/EMT experiment were
replayed from the ORIGINAL start of the failed Gaussian stage: saved center plus
width times direction. Every Gaussian center, direction, width and height was
held fixed to its archived value. No failed endpoint was used as a favorable
restart, and no original failure was relabeled successful.

The two methods are installed ASE LBFGS and LBFGSLineSearch, otherwise using their
unmodified defaults. Both have at most 201 physical-surface E/F requests and 200
accepted optimizer steps, fmax=0.01 eV/Angstrom. A request cap can interrupt a
line search; only completed accepted iterates are eligible for the force check.
A budget-interrupted trial is not called converged. The experiment selected all
31 failures, not a favorable subset. It does not estimate performance over all
SSW stages, because originally successful stages were not included.

Source: `research/ga_ssw/compare_failed_quench_linesearch.py`.
Evidence: `research/ga_ssw/evidence/cu13-failed-quench-linesearch/` contains the
pre-run plan, launch script, installed ASE optimizer source, all accepted-iterate
E/F/coordinates/history diagnostics, and complete summary.

## Findings

| Method | Modified quenches converged / 31 | E/F requests | Runs with negative curvature history pairs | Runs with uphill directions |
|---|---:|---:|---:|---:|
| LBFGS | 0 | 6,231 | 3 | 3 |
| LBFGSLineSearch | 7 | 5,479 | 0 | 0 |

All 31 ordinary-LBFGS final geometries exactly reproduce the archived failed
endpoints (maximum coordinate difference zero). Thus the frozen subproblem and
start reconstruction is supported by execution, not just matching total force.
The other 24 line-search runs exhaust their E/F caps. The whole additional
experiment costs 11,710 E/F requests; its outcomes are separate from the original
60-move search denominator.

The seven converged line-search cases are seed17/dimer steps2,4,14;
seed17/Ritz steps2,10,14; seed3/Ritz step8 (indices as archived). These are only
converged MODIFIED-surface quenches, not certified true minima, accepted SSW
moves, or new successful full-search trajectories.

## Interpretation

The installed LBFGS state stores rho=1/(s dot y) without rejecting negative
curvature pairs; without line search its step is bounded geometrically rather
than accepted by an objective-decrease test. Uphill-step diagnostics use the
previous accepted gradient dotted with the actual next displacement. Negative
curvature and uphill steps occur in only three of the 31 failures, so neither
explains the entire failure set. A line-search replacement improves some frozen
subproblems, but it does not solve the dominant remaining budget exhaustion.

The mathematical distinction between sufficient decrease, curvature conditions
and a descent direction is standard; see the primary
[SciPy line-search documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html).
That documentation supplies theory context, not a claim that ASE uses SciPy's
implementation or that its default numerical constants are identical. The actual
ASE class implementation is archived alongside this experiment.

The experiment rules out two overly simple explanations: these failed endpoints
are not merely unremoved rigid force (prior force decomposition), and not all
failures are caused by a non-descent L-BFGS history. It does not yet isolate
ill-conditioning, the optimizer metric, or the interaction of multiple Gaussians
and stage termination. No new heuristic, restart, enlarged cap or temperature
change has been introduced into the independent driver.

Next compare convergence histories/conditioning and original optimizer semantics
before choosing a local optimizer policy. Any full SSW comparison must include
all originally successful stages, line-search trial evaluations, final true
quenches and failures under a prospectively fixed cost budget. Do not promote
line search based solely on seven recoveries in a failure-conditioned sample.

## Native optimizer follow-up changes the implementation assessment

The bounded static follow-up in `native-local-optimizer-linesearch.md` resolves
an actual BFGSDRIVER -> LBFGS -> MCSRCH -> MCSTEP call chain. Thus choosing ASE
LBFGS without line search was an explicit numerical substitution, not native
local-optimizer parity. The native settings must still be recovered separately:
its ELF GTOL initializes to 900, and the inspected path does not reset that large
value to 0.9. Full-program runtime overwrites have not been excluded. Do not call
this a universally standard strong-Wolfe configuration, and do not infer that
switching to ASE LBFGSLineSearch reproduces it. FTOL and step bounds come from
caller parameters; history scaling also differs. Native arithmetic anomalies
remain separately named research behavior rather than production defaults.
