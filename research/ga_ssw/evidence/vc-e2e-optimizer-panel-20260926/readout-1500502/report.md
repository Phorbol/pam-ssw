# VC end-to-end optimizer panel: artifact analysis

Read-only analysis of the supplied worker artifacts; no calculator/PES calls were made. A fresh-valid endpoint is a separate EFS recheck passing the configured numeric certificate. Algorithm qualification and fresh qualification are reported independently.

| Case | Seed | Method | Exit | Status / algorithm | Outer steps | Landing events | No landing | Algorithm-qualified / fresh-valid | Search / fresh requests | Censor |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| AlOH26 | 71 | ase | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 3212 / 4 | — |
| AlOH26 | 71 | safe_total | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 875 / 4 | — |
| AlOH26 | 71 | scipy | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 1000 / 4 | — |
| AlOH26 | 83 | ase | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 2481 / 4 | — |
| AlOH26 | 83 | safe_total | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 2275 / 4 | — |
| AlOH26 | 83 | scipy | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 2820 / 4 | — |
| TiO2-phase87 | 71 | ase | 0 | completed / completed | 3 | 2 | 1 | 2 / 2 | 3514 / 3 | — |
| TiO2-phase87 | 71 | safe_total | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 2767 / 4 | — |
| TiO2-phase87 | 71 | scipy | 0 | completed / completed | 3 | 2 | 1 | 2 / 2 | 3164 / 3 | — |
| TiO2-phase87 | 83 | ase | 0 | completed / completed | 3 | 2 | 1 | 2 / 2 | 5813 / 3 | — |
| TiO2-phase87 | 83 | safe_total | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 3645 / 4 | — |
| TiO2-phase87 | 83 | scipy | 0 | completed / completed | 3 | 3 | 0 | 3 / 3 | 4387 / 4 | — |

## Paired seeds by case and method

| Case | Seed | Method | Outer steps | No landing | Landing events | Algorithm qualification failures | Fresh qualification failures | Accepted / MC-rejected | Tight / broad distinct groups |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| AlOH26 | 71 | ase | 3 | 0 | 3 | 0 | 0 | 3 / 0 | 3 / 3 |
| AlOH26 | 71 | safe_total | 3 | 0 | 3 | 0 | 0 | 3 / 0 | 3 / 3 |
| AlOH26 | 71 | scipy | 3 | 0 | 3 | 0 | 0 | 3 / 0 | 3 / 3 |
| AlOH26 | 83 | ase | 3 | 0 | 3 | 0 | 0 | 2 / 1 | 3 / 3 |
| AlOH26 | 83 | safe_total | 3 | 0 | 3 | 0 | 0 | 1 / 2 | 3 / 3 |
| AlOH26 | 83 | scipy | 3 | 0 | 3 | 0 | 0 | 1 / 2 | 3 / 2 |
| TiO2-phase87 | 71 | ase | 3 | 1 | 2 | 0 | 0 | 1 / 1 | 2 / 2 |
| TiO2-phase87 | 71 | safe_total | 3 | 0 | 3 | 0 | 0 | 1 / 2 | 3 / 3 |
| TiO2-phase87 | 71 | scipy | 3 | 1 | 2 | 0 | 0 | 1 / 1 | 2 / 2 |
| TiO2-phase87 | 83 | ase | 3 | 1 | 2 | 0 | 0 | 0 / 2 | 2 / 2 |
| TiO2-phase87 | 83 | safe_total | 3 | 0 | 3 | 0 | 0 | 0 / 3 | 3 / 3 |
| TiO2-phase87 | 83 | scipy | 3 | 0 | 3 | 0 | 0 | 0 / 3 | 3 / 3 |

## Cost, stages, and interpretation

Across artifacts, charged search requests: 35953 known; fresh recheck requests: 45 known. Incomplete/missing arms: 0 of 12; their unknown costs are not treated as zero. Plan ceilings are 6000 search and 4 fresh requests per arm (72000 and 48 total for the full panel).

Denominators: 36 recorded outer steps across 12 arms; 3 outer steps produced no landing and 33 did. There were 45 independent fresh checks: 12 initial structures plus landing events. Initial qualification is reported separately per arm. A no-landing outer failure is not counted as a landing qualification failure or MC rejection.

No-landing outer records: TiO2-phase87 seed71 ase step2=biased_quench_failed; TiO2-phase87 seed71 scipy step2=biased_quench_failed; TiO2-phase87 seed83 ase step2=biased_quench_failed.

Stage sums use charged surface request counts: the initial quench plus its certification; climb rotation, height, biased-quench, and true-check counters; and each outer record's residual as post-climb true-quench plus certification. `landing_optimizer.requests` is retained as a descriptive attempted-call counter only, since it can include a denied budget call. Any negative post-climb residual is retained with an accounting warning. The arm-level residual `search_requests - stage sum` and ledger-to-summary deltas expose remaining discrepancies. The worker ledger has no stage labels, so it reconciles charged totals rather than imputing stages.

Structure matching uses pymatgen StructureMatcher with the prior VC pilot's tight (ltol 0.05, stol 0.10, angle 2°) and broad (0.20, 0.30, 5°) settings, `scale=False`, `primitive_cell=False`, `attempt_supercell=False`, and ElementComparator. It compares each arm's initial plus fresh-valid, algorithm-qualified new landings. Group counts are greedy summaries with full pairwise matrices retained; approximate matches are not basin, phase, or global-minimum identities. Energy/objective and volume changes are reported relative to the separately fresh-checked initial structure.

The configured force/stress thresholds are numeric local certificates on this MACE OMAT model PES. This small two-seed panel can describe endpoint qualification and cost for these inputs; it cannot establish stable optimizer benefit, global search efficiency, phase identity, or transfer to other systems.

Analysis elapsed 3.807 s; calculator/PES calls: zero.
