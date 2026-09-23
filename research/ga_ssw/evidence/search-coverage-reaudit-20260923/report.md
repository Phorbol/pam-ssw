# Partial readout: fixed-cell geometry re-audit

The single 5-minute CPU job timed out after processing 3 of the 12 planned periodic arms; the remaining 9/12 were not analyzed. These are retrospective views of previously completed runs, not new independent searches. There is no AlOH result and no brookite BH seed29 result. No retry or expanded allocation was submitted.

| Arm | Stored frames | Tight groups | Broad groups | Frames matching initial (all) | Later recurrence, excluding first frame |
|---|---:|---:|---:|---:|---:|
| brookite48 SSW seed11 | 9 | 2 | 2 | 8/9 (88.9%) | 7/8 (87.5%) |
| brookite48 SSW seed29 | 9 | 2 | 2 | 8/9 (88.9%) | 7/8 (87.5%) |
| brookite48 BH seed11 | 62 | 54 | 54 | 8/62 (12.9%) | 7/61 (11.5%) |

Tight and broad StructureMatcher tolerances gave the same group and initial-return counts for these three arms. The SSW runs each saved nine minima frames but only two representative groups, with seven of the eight later frames matching the initial quenched structure. This supports the narrow interpretation that saved-frame count overstates geometric coverage for these two trajectories. The BH seed11 arm also contains duplicates (62 frames versus 54 representative groups).

These groups are order-sensitive geometric clusters under the protocol, not strict basin labels. In particular, the 54 BH groups do not establish 54 distinct basins or better low-energy coverage. This partial sample does not support method ranking: it contains two SSW seeds and one BH seed from one system, and all arms reuse the original search inputs and trajectories.

The job ran as Slurm 1467849 on CPU-MISC/rush-cpu, account `sjtu-caoxiaoming`, using `mace_env` Python. `sacct` reports TIMEOUT at 00:05:13, 2 scheduler-assigned CPUs, and MaxRSS 120772K. The 00:10:26 figure is allocated core time (scheduler-assigned cores multiplied by elapsed time), not measured busy CPU time. The geometry analysis made no PES/model calls.
