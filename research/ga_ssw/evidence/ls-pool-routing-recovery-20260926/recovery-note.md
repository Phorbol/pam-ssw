# LS pool-routing interrupted-arm recovery

This is a derived, read-only recovery of array 1500677's C60-isomer2/MC arm. The original run directory under `c60-local-defect-qualification/.../ls-pool-routing-20260926/run-1500677-3` was not modified. `recover_mc_summary.py` documents the sidecar derivation from raw artifacts. Its output is `derived-view/C60-isomer2/mc/summary.json`; the analysis inputs are linked read-only under `derived-view/`.

The snapshot identifies worker commit `6703f3511226e6ba99ef83279c8cec983204990a` and an exact copied `worker.py` SHA256 match (`123612e948f634d4c4019f0244a28061133a53074261e87a8774ae9b3355c76b`). The arm has a search wall budget of 1700 s within a 1800 s process budget. Its saved `search-result.json` and checkpoint precede fresh validation; the fresh files then close all candidates. The expected post-fresh artifacts `offline-identity.json`, `minima.extxyz`, a rewritten `search-result.json`, and `summary.json` are absent. In the copied worker source, MC identity summarization precedes those writes. This ordering makes interruption during post-search identity summarization the leading explanation, but the artifacts alone do not prove the scheduler-level cause.

## Cost and coverage closure

- Search ledger: 28,338 charged completions with contiguous request IDs 1–28,338, then one uncharged `wall_deadline` censor. Initial plus all 66 outer-record request counts equals both `result.evaluation_requests` and the wrapper's 28,338 requests.
- Fresh ledger: all 66 candidates have charged completions with contiguous IDs 1–66. The 66 checks map to the initial structure plus all 65 returned landings; every check is `fresh_completed` and certified. There are 66/66 fresh-qualified candidates.
- Recovered charged cost: 28,338 search + 66 fresh = 28,404 E/F requests. This arm is still censored at 66/100 outer records; it is not a completed 100-step run.
- In the exact three-mode C60 common search prefix of 23,877 requests, fresh-certified converged minima counts are MC 55, uniform 61, and PAM 61. This one-seed prefix is a bounded observation, not a controller ranking.

The authoritative analyzer result is the CPU-MISC output from job 1501026: `readout-cpu-1501026/report.md`, with the job result in `cpu-1501026.out`. The command and environment are preserved in the work record `verify-cpu.sbatch`; the run used analyzer source at commit `59c85f7`. It reported all six arms censored by their wall deadline before 100 outer records, but all ledgers and fresh candidates closed. Total paid cost was 180,478 search plus 366 fresh E/F calls. The exact common prefixes were 33,031 for C4H6 and 23,877 for C60-isomer2.

The MC summary sidecar can be independently re-derived without calculator imports or raw writes:

```bash
python research/ga_ssw/evidence/ls-pool-routing-recovery-20260926/recover_mc_summary.py \
  --raw-arm /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/ls-pool-routing-20260926/run-1500677-3 \
  --output /tmp/ls-pool-mc-recovered-summary.json
```

Array input mapping is `case_index * len(plan.methods) + mode_index`, with the plan's case and method order. The recovery script requires the raw summary to remain absent and refuses to write inside the raw arm or overwrite its destination.

Two earlier local analyzer outputs, `readout-1500680-final/` and `readout-1500680-authoritative/`, are preliminary manual runs, not scheduler-qualified artifacts. They are retained for traceability; use CPU1501026's report as the authoritative readout.

The authoritative job command was:

```bash
sbatch research/ga_ssw/evidence/ls-pool-routing-recovery-20260926/verify-cpu.sbatch
```

The CPU report retains the censored statuses and separates search/fresh costs. The C60 graph classifications and energy-window counts are separate. At all three fixed cutoffs (1.64, 1.70, 1.80 Å), the prefix contains 12/3/4 cage-candidate observations for MC/uniform/PAM; source-defect matches are 12/2/3; Ih graph matches are 0/0/1. The single PAM Ih hit is candidate/minimum 54, outer record 53, first observed at 21,055 cumulative paid search requests; it was accepted and freshly certified at -62215.39289098097 eV, fmax 0.0339136 eV/Å. It is a 90-edge cage and matches the Ih graph at all three cutoffs, but not the source-defect graph. The 0.01 eV IH energy-window count is 0/0/1; the PAM hit is about 0.000480 eV above the reference. No committed selector restart occurred on that record or the immediately preceding record.

C4H6 connected topology representatives at the common 33,031-request prefix have no H degree greater than one and no C degree greater than four under the frozen graph cutoffs. Their minimum pair distances span 1.074–1.082 Å; maximum representative energy offsets from the lowest fresh connected observation are 3.199, 3.200, and 1.047 eV for MC/uniform/PAM. These were local, preliminary offline geometry diagnostics, not part of CPU1501026, and do not establish chemical validity. See `c4h6-connected-class-sanity.json` and `c60-pam-ih-hit.json`.
