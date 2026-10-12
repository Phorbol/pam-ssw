# C60 source-3 continuation: run-v1 summary

Four arms resumed from the immutable parent checkpoints and reached cumulative outer index 100: SSW and paper-LS at seeds 26100791 and 26100792. The source was frozen at `4fec3e3f7d09009f0808dfc43cc1c2c2e78900d6`; the target calculator was MACE-MH-1, `omol`, float64 CUDA. This is a target-model continuation, not a PBE calculation or reproduction of the C60 paper's G-NN potential/search statistics.

| Seed | Arm | Parent outer / requests | Child outer / requests | Cumulative requests | Child wall (s) | Best connected energy at 100 (eV) | Fresh E/F numerical check | Parent 1.8 Å cage + energy target | Strict Ih target |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 26100791 | SSW | 37 / 15,791 | 63 / 28,105 | 43,896 | 1,618 | -62,213.551234919 | pass | no | no |
| 26100791 | paper-LS | 37 / 15,917 | 63 / 27,352 | 43,269 | 1,555 | -62,213.551280972 | pass | no | no |
| 26100792 | SSW | 35 / 15,615 | 65 / 28,703 | 44,318 | 1,656 | -62,213.551311739 | pass | no | no |
| 26100792 | paper-LS | 34 / 15,754 | 66 / 28,967 | 44,721 | 1,655 | -62,213.551264364 | pass | no | no |

All arms ended with `algorithm_status=completed`, `cumulative_outer_count=100`, no exception, no request denial, and no uncheckpointed paid tail. Child request ledgers are sequential, contain no failed E/F entries, and close exactly against checkpoint lineage cost. Each parent checkpoint is byte-identical to its `checkpoint-start.pkl`; the parent plan, budget, and request-ledger hashes were unchanged. Across the four arms, parent search cost was 63,077 requests, child search cost 113,127, and combined search cost 176,204. Parent fresh checks total 8; child fresh checks total 4. Child execution times sum to about 6,484 seconds; each arm remained within its 30-minute allocation. The 60,000 cumulative-request prefix was not reached by any arm.

All four child fresh checks reproduced the search energy exactly, passed the configured force threshold, and had `numerical_qualified=true`. Each was a 1.8 Å graph-cage candidate, so the existing manual `physical_cage_review` remains required. None was within the 0.01 eV reference-energy window; none matched the Ih reference graph at 1.64, 1.70, and 1.80 Å. The parent fresh checks likewise had no joint target. Across the complete parent-plus-child observation curves, each arm had multiple converged 1.8 Å cage candidates but zero cage-plus-energy joint candidates and zero strict Ih target candidates.

The paired fixed-cost best-connected-energy comparison changes sign across the two repeated seeds: at 100, paper-LS is lower than SSW by approximately 0.000046 eV for seed 26100791, while SSW is lower than paper-LS by approximately 0.000047 eV for seed 26100792. These differences are tiny relative to the roughly 1.842 eV remaining to the frozen reference energy. The per-arm and paired cumulative curves, cutoff graph diagnostics, LS costs, and MC/status counts are in `cumulative-readout.json` and `paired-cumulative-readout.json` under each run directory.

The reported graph-isomorphism class counts (70–79 at 1.70 Å) are graph descriptors only. They do not establish distinct geometric basins, basin coverage, or global exploration success.
