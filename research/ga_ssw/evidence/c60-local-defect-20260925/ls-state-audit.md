# Native-LS state and request-cost audit

The native-LS switch was active in the recorded trajectories. All 20 native-LS outer records contain the `normal_update` action; the C–C table coefficient changes across steps, and each update has `q=1`. The configured response target was 20 meV/atom. The saved true-PES prequench response is 3.4638–8.1179 meV/atom across the 20 records (17.3–40.6% of target), so every measured response was positive but below the target.

The configured `ratio=1.1` and `cycle=100` yield `nsoftstep=110`; the periodic-cycle predicate `0 < nsoftstep < cycle` is therefore false. This disables only the periodic save/zero/restore cycle. `lselfadapt=true` remains set, and `normal_update_due` independently updates every nonzero step through `presteps=100`; the result records confirm that path ran. The controlling predicate and separate cycle/adaptive branches are in [native_ls.py](/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/pamssw/standalone/native_ls.py:189). The observed updated C–C coefficient spans 0.07034137–0.10653914 eV; including the initializer value (0.0666666627 eV, from the documented normalization with N=60 and 90 cage bonds), the range is 0.06666666–0.10653914 eV.

| Seed | Method | Initial quench | LS preparation | Climb | True quench | Total requests |
|---:|---|---:|---:|---:|---:|---:|
| 1101 | ssw_without_ls | 1 | 0 | 5581 | 738 | 6320 |
| 1101 | native_ls | 1 | 71 | 5478 | 837 | 6387 |
| 1102 | ssw_without_ls | 1 | 0 | 6266 | 637 | 6904 |
| 1102 | native_ls | 1 | 70 | 6126 | 701 | 6898 |

Each row sums exactly to the `evaluation_requests` total in its `result.json`. Climb is the arithmetic remainder after LS preparation and true-quench landing calls are removed from the record totals. Paired total-request differences (native LS minus SSW) are +67 for seed 1101 and −6 for seed 1102. The two seeds are repeated probes of one starting defect; this accounting does not establish a general cost or success advantage.

The numeric source and calculations are in [ls-state-audit.json](ls-state-audit.json). No calculations were rerun.
