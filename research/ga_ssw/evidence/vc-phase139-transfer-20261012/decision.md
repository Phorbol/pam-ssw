# Phase139 transfer: close the atomic-contribution panel without tuning

2026-10-12. Development input transfer, not an independent success-rate estimate.
Runner/core provenance is frozen in run-1757298/manifest.json (628848d), and the
source inputs/entrypoint/helpers are retained in its source-snapshot directory.
[Protocol](protocol.md); [readout](readout-1757298.json). No core/default changes.

## Result and qualification

| Seed | Arm | Paid / actual search EFS | Valid landings | First cold-confirmed anatase |
|---|---|---:|---:|---:|
|26101251|cell_only|4000 / 3967|20|2767 requests|
|26101251|cell_plus_atomic|4000 / 3896|7|none before cap|
|26101252|cell_only|4000 / 3968|19|1284 requests|
|26101252|cell_plus_atomic|4000 / 3888|7|none before cap|

All initial quenches are force/stress-qualified and non-target. All selected
fresh endpoints meet the existing force/stress gates and reproduce their search
energies. The seed51 first target is also the best frame, so its independent
check is recorded as best_physical; it is not missing a cold check. Seed52 has a
separate first_target cold check. Both identity tolerances and the frozen target
energy window pass for both first arrivals. No relaxed historical input
certificate was reused for these freshly quenched raw inputs.

The atomic arms complete 30 and 33 Gaussians, respectively; all 63 rotations
use the explicit pre-existing finite-budget release. Six atomic-stage valid
landings (three each) are all uphill, MC-rejected and non-target. The best
force/stress-qualified energy in both atomic arms remains the initial energy,
whereas both cell-only best frames qualify as anatase. Thus this transfer has
no energy counterexample favoring the atomic treatment. The earlier phase87
panel did have mixed best-energy rankings; that contrary evidence remains.

## Costs and boundaries

Total 16000 paid search requests, 15719 actual model calls, plus 10 fresh/reference
calls. Raw search and fresh ledgers close against per-arm records; no paid
calculator failure. Three arms have one budget denial each. The seed51 cell-only
arm completes its twentieth valid landing exactly at 4000 paid requests without
a denial, so it is completed rather than budget-failed. The two atomic arms
retain 24 zero-request post-cap records; these are not additional attempts.
Atomic stage accounting leaves 30/75 requests unassigned; total costs include
them, and stage subtotals are not claimed exhaustive. Campaign artifact wall
is 229.55 s, not an isolated optimizer timing experiment.

Shared initial, first outer and pre-atomic cell traces agree within the existing
trace diagnostic in both pairs. This does not make later random-number streams
identical: the atomic treatment consumes additional random numbers. Across
phase87 and phase139, seeds differ, so no pure initial-geometry causal contrast
is claimed. The two inputs give four cell-only target arrivals versus one in
four atomic-treatment runs at these bounds, but four seeds of two TiO2 starts
are not a general success-rate estimate or evidence against atomic SSW in other
materials/regimes. The atom-containing proposals are more expensive and permit
fewer outer attempts at the same paid budget; that is part of this efficiency
comparison, not an equal-outer-step comparison.

## Decision

Do not promote, remove or retune the optional atomic block, and do not extend
these input trajectories or add seeds to manufacture a preferred result. The
bounded question is answered: enabling atomic climbing is not a missing
execution capability and has not improved target arrival on either tested
input. Preserve the simpler cell-only reference for these inputs. The next
useful comparison prerequisite is the native external stress contract, because
the existing LASP/ASE bridge only supplies E/F and rejects changing cells. A
same-OMAT native VC baseline is not yet available; native file acceptance with
an omitted tensor cannot establish one. That prerequisite is a bounded
interface diagnostic, not a new search strategy.

Reproduce analysis (zero new PES):
`python research/ga_ssw/evidence/vc-atomic-contribution-20261012/summarize.py research/ga_ssw/evidence/vc-phase139-transfer-20261012/run-1757298 --output NEW_READOUT.json`.
Eight targeted contract checks, zero-PES preflight, shell syntax and diff checks
passed before submission. These checks qualify the runner, not scientific
performance. Raw run/log files remain outside Git; this compact readout and
protocol locate them without overwriting or filtering failures.
