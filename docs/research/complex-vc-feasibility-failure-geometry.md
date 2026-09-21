# Complex VC feasibility: failure stages and geometry

All **8/8 runs** are terminal: AlOH26 and brookite48, seed 17, four arms, two planned outer slots each. This is a single-seed numerical-feasibility experiment, not an efficiency campaign. Total search cost is **15,467 E/F(/stress) requests**, including **436 initialization requests**; recorded run wall times sum to **5,217.735 s**. This audit adds **0 PES requests** and does not use the concurrently running Hessian qualification as completed evidence.

| System | Arm | Initial | Outer 0 | Outer 1 | Total | Valid new landings (accepted/rejected) |
|---|---|---:|---:|---:|---:|---:|
| AlOH26 | fixed | 99 | 1359 | 542 | 2000 | 0 / 1 |
| AlOH26 | posterior cell | 99 | 1424 | 477 | 2000 | 0 / 1 |
| AlOH26 | block VC | 99 | 236 | 1132 | 1467 | 0 / 2 |
| AlOH26 | joint VC | 99 | 1327 | 574 | 2000 | 1 / 0 |
| brookite48 | fixed | 10 | 1185 | 805 | 2000 | 0 / 1 |
| brookite48 | posterior cell | 10 | 1195 | 795 | 2000 | 0 / 1 |
| brookite48 | block VC | 10 | 189 | 1801 | 2000 | 0 / 1 |
| brookite48 | joint VC | 10 | 1990 | 0 | 2000 | 0 / 0 |

All 16 planned slots remain in the denominator: 8 valid landings (1 accepted, 7 rejected), 7 actual budget-censored proposals, and one zero-request slot after exhaustion. The previous offline summary labeled brookite joint outer 0 `failed` and outer 1 `budget_censored`; the frozen raw labels remain unchanged, but the corrected derived classification is **one paid truncated attempt and one unfunded slot**, not two independent failed searches.

## Why brookite joint has no new landing

Initialization converged in 10 EFS. The first outer proposal used all remaining 1990 EFS. Gaussian stages 0–8 report converged biased optimizations; their true objectives range from -424.2225 to -420.0725 eV, all above the initial -427.60555 eV. Stage 9 reports `evaluation_failed`; its direction solve had already returned a finite residual 0.01782, below the configured rotation tolerance 0.02. The exact 2000 cap was reached. The next outer slot immediately reports `declared material comparison budget exhausted` with **zero** new requests.

Thus the observed bottleneck is the cost of completing the conservative multi-Gaussian climb under this budget. There is no observed true quench for this proposal. It would be incorrect to say the endpoint quench failed, that direction refinement never converged, or that this proves intrinsic optimizer instability. The frozen runner's stage records lack per-Gaussian EFS and detailed failed-optimizer trace, so rotation versus line-search cost cannot be apportioned exactly without another run; none was launched.

Reconstructing the saved q/cell states with the documented symmetric log-strain map gives volume 525.973 Å³ initially, 548.170–697.962 Å³ over the first nine converged biased stages and 699.228 Å³ at the saved last stage (~33% larger than initial). Minimum MIC pair distances over saved states are 1.636–1.764 Å, versus initial 1.868 Å. The final cell singular values are 16.442, 8.287 and 5.132 Å (initial 13.081, 7.766, 5.178 Å). This is a substantial biased structural/cell deformation with shorter contacts, not a pure rigid motion. Positive nonsingular cells do not establish physical validity; these are biased states, not certified new phases. Nor do these few saved points prove that expansion caused the request exhaustion.

## What changed in the accepted AlOH joint candidate

The first joint proposal completed six biased stages and a true quench in 1327 EFS after initialization. Its energy is **-177.273254065 eV**, versus **-177.072668602 eV** initially: **ΔE = -0.200585463 eV** at zero pressure. Recorded landing residuals are fmax 0.0099203 eV/Å and maximum stress 2.97e-5 eV/Å³. Volume decreases from **273.7803 to 268.7220 Å³** (1.85%). These are the search's numerical force/stress certificates, not a Hessian stability or new-phase certificate.

Using original zero-based atom indices and exact minimum-image distances:

- **H8 changes nearest oxygen from O2 (0.9819 Å) to O12 (0.9906 Å).**
- **H20 changes nearest oxygen from O16 (0.9814 Å) to O19 (1.0152 Å).**
- H4 remains nearest O1; its second oxygen changes O11 (1.6348 Å) to O7 (1.4146 Å).
- H18 remains nearest O15; its second O19 distance shortens 2.0682 → 1.3995 Å, while its nearest O–H elongates 0.9827 → 1.0759 Å.

Every H retains exactly one O neighbor at cutoffs 1.1, 1.2 and 1.3 Å. These endpoint data support changed proton attachment and altered secondary O contacts, not isolated free H. They do not establish an elementary proton-transfer path, its chronology or barrier.

Al–O connectivity changes throughout the framework. At a descriptive 2.2 Å cutoff, Al3 exchanges O12 for O13; Al6 loses O0/O7 and gains O2/O24; Al9 loses O7 and gains O17/O23; Al10 exchanges O1 for O11; Al14 loses O13/O23 and gains O0; Al21 exchanges O19 for O15; Al22 exchanges O24 for O16; Al25 exchanges O17 for O1. Coordination-count sensitivity at cutoffs 2.0/2.2/2.4 Å is retained in the JSON: Al9 changes 3/3/3 → 3/4/4, Al14 changes 4/4/4 → 3/3/3, Al25 changes 3/3/4 → 3/3/3. Other Al counts remain constant while oxygen identities change. These cutoffs are reporting sweeps, not fitted bond-order rules or algorithm parameters.

The candidate therefore involves both hydrogen rearrangement and Al–O network reorganization; interpreting it as merely an elastic cell improvement is unsupported. MACE OMAT-small is the actual oracle. Its prediction here is not automatically a validated chemical reaction or first-principles phase stability. Higher-level model checks and the separate qualification remain distinct tasks.

## Other seven paid failure/censoring exits and evidence

AlOH fixed/posterior terminate their second proposal in atomic climbing after 542/477 calls; AlOH joint terminates its second climb in Gaussian stage 4 after 574 calls (four earlier biased stages converged). Brookite fixed/posterior terminate second atomic climbing after 805/795 calls. Brookite block reaches second-proposal true quench but the 1801-call remainder is insufficient for a certified landing. AlOH block completes both proposals, but both are higher-energy and rejected (+0.35515 and +1.48889 eV). All rejected landings remain included in the table.

Source results: `research/ga_ssw/prospective/complex-vc-feasibility/results/*/result.json`; derived offline classification corrected using explicit BudgetExhausted ledger entries and cumulative initialization/outer request intervals. Reproducible zero-PES audit: `research/ga_ssw/analyze_complex_vc_failure_geometry.py`; output `research/ga_ssw/evidence/complex-vc-failure-geometry/analysis.json`. No frozen source or search result was modified.

The pure offline `material_budget_outcomes.py` maps each event to its paid [start,end] request interval and retains the explicit BudgetExhausted ledger evidence. No run-level `censored` flag or vague substring is used to infer a failed event. `tests/research/test_material_budget_outcomes.py` verifies all 8 actual run artifacts, both slots each, total 15467 and exact 1 accepted / 7 rejected / 7 censored / 1 not-started counts: **1 passed**. The geometry/matcher analyzer was rerun with 0 EFS; only derived offline outputs changed.
