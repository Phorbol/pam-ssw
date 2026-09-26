# Periodic direction pilot: approximate structure coverage

Geometry-only comparison of each arm’s initial minimum and its three true landing minima. Matching uses the prior periodic rotation audit’s tight and broad tolerance sets, with fixed cell scale, primitive-cell, supercell, and element-comparison settings. Energy values do not enter matching.

| Case | Arm | Search E/F requests | Fresh endpoints | Tight groups incl. initial | Tight returns | Broad groups incl. initial | Broad returns | Local replay requests / equal state |
|---|---|---:|---:|---:|---:|---:|---:|---|
| rutile | global | 1259 | 4 / 4 | 3 | 1 / 3 | 3 | 1 / 3 | — |
| rutile | local_memory | 1032 | 4 / 4 | 2 | 2 / 3 | 2 | 2 / 3 | 1032 / True |
| anatase | global | 1405 | 4 / 4 | 4 | 0 / 3 | 4 | 0 / 3 | — |
| anatase | local_memory | 1067 | 4 / 4 | 1 | 3 / 3 | 1 | 3 / 3 | 1067 / True |
| brookite48 | global | 1251 | 4 / 4 | 2 | 2 / 3 | 2 | 2 / 3 | — |
| brookite48 | local_memory | 929 | 4 / 4 | 1 | 3 / 3 | 1 | 3 / 3 | 929 / True |

The groups are sequential representative clusters over `initial, landing_0, landing_1, landing_2`; the JSON retains the full pairwise matrix and representative assignments at each tolerance. Because approximate matching may not be transitive, group counts are order-dependent summaries, not unique basin counts.

Cross-arm landing-to-landing pair matches:

| Case | Tolerance | Matching global/local landing pairs |
|---|---|---:|
| rutile | tight | 2 / 9 |
| rutile | broad | 2 / 9 |
| anatase | tight | 0 / 9 |
| anatase | broad | 0 / 9 |
| brookite48 | tight | 6 / 9 |
| brookite48 | broad | 6 / 9 |

Across these particular runs, search totals were 3915 requests for the global arm and 3028 for local memory (6943 combined). This is a raw cost observation from one seed and three steps per case, not an efficiency estimate.

Tight and broad pairwise relations were identical for all six within-arm panels: `True`. The match/return and grouping readout was insensitive to this tolerance change for these sampled frames.

All six runs completed three outer steps. Search request totals include the initial quench and all recorded steps; fresh endpoint checks are four separate recalculations per arm. The local-memory arm’s replay uses the saved oracle stream and is a same-trajectory resume check, not independent sampling.

A landing matching the initial structure is counted as a return under the stated tolerance. Nonmatches show geometric difference under that matcher only; neither result certifies a distinct basin, positive Hessian, or a new phase. The three-step, three-case panel is a development pilot and supports no universal efficiency claim.

Matcher: pymatgen 2026.5.4; analysis elapsed 3.599 s; calculator/PES calls: zero.
