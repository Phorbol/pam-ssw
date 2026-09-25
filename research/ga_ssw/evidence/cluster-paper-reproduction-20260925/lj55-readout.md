# LJ55 development result, not published success-rate replication

Two independent random-coordinate inputs, each shared by the two frozen SSW direction settings and plain ASE BH. Published target -279.248470epsilon is independently qualified. All four SSW energy candidates passed fresh energy/force checks and proper-rotation/permutation geometry matching (CPU1488492 closes the fourth geometry). No target structure was used as their starting geometry.

| Method | Seed | First target search requests | Best energy/epsilon | Outcome |
|---|---:|---:|---:|---|
| SSW global direction |25092501|54139|-279.248455|target found|
| SSW global direction |25092502|71555|-279.248451|target found|
| SSW paper pair+global direction |25092501|44661|-279.248464|target found|
| SSW paper pair+global direction |25092502|129724|-279.248465|target found|
| Plain ASE3.26 BH |25092501|not reached within200000|-273.505406|request-censored|
| Plain ASE3.26 BH |25092502|not reached within200000|-276.603173|request-censored|

BH CPU1488453 used400000search+2fresh requests in438.67s. Both best endpoints pass fmax0.01eV/Angstrom; neither matches the target energy. SSW uses8fresh requests for these four runs. Published random-initialization distribution is unspecified and published samples are much larger, so this is target-discovery reproduction under our explicit initialization, not replication of reported success rates. Pair-direction cost improves one seed and worsens the other; no default promotion. These outcomes establish a useful positive control for the present independent SSW implementation, not a universal advantage over BH or LASP.

## Baseline scope and accounting audit

Plain ASE3.26 retains its accepted raw-proposal coordinates as the next displacement origin; SSW retains an accepted quenched minimum. ASE source `optimize/basin.py` lines104-107 and146-158 distinguish those states. The runner deliberately does not patch the library. The BH1997 adaptive displacement, container, angular operations and neighboring-size seeds are absent; see `lj55-bh-plan.md`. This is a mature-library operational baseline with a common Safe-total quench, not the complete1997 algorithm.

BH post-quench `get_value()` makes one additional surface request after each successful optimizer return. It is included in the bounded surface and final summary, but lies after the corresponding quench JSONL row. Therefore those rows describe optimizer costs, not complete BH-step costs; do not plot them as complete-step cumulative costs without accounting for the subsequent read. First-hit exception stops before this read. Total request caps and final budgets include it. Initial quench is performed once and included by both methods. The audit found no basis to rerun, change displacement, or tune this baseline from these two outcomes.

LJ38 remains a separate hard-case readout. Wall-censored trajectories cannot establish an equal-force-budget failure rate, and the measured history-copy overhead must not be inferred to explain all their runtime without a matched profile. The approved compact observer is an engineering correction with separate equivalence checks.
