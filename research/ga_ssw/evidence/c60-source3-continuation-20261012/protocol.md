# C60 source-3 paired continuation protocol

## Question and scope

Complete the already frozen 100-outer-attempt SSW versus paper-LS comparison on the published C60 isomer-3 input. This is a continuation of the two existing search seeds, not a new seed campaign. It tests the incremental effect of the existing LS arm over this fixed horizon; it cannot establish a general barrier explanation or reproduce the paper's PBE results. The published C60 defect geometry and the paper's G-NN surface are source context; this experiment instead uses the already frozen MACE-MH-1 target model. It is neither a PBE result nor a reproduction of the original potential or search statistics.

The immutable parent is `/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/prepared-20261007-a`. Its core snapshot, source plan, model (`MACE-MH-1`, `omol`, float64 CUDA), nonperiodic 50 Å C60 input, SSW configuration, LS parameters, and seeds remain authoritative. The child reads the parent and copies only each safe outer-boundary checkpoint into a new output directory. It never changes parent files.

The public checkpoint contract restores current and best structures, minima, completed records, the RNG bit-generator state, native Monte Carlo state, and LS frozen-bond/response state. `steps` means additional attempts, so the driver computes `100 - checkpoint.next_index`. It uses the same frozen `pamssw` package imported from the parent's prepared snapshot. No initial quench is repeated.

## Fixed factors and limits

The four paired trajectories are seed 26100791 SSW/paper-LS and seed 26100792 SSW/paper-LS. All model and search parameters are inherited exactly from the parent plan. The only extension is continuation from each saved state to cumulative outer index 100.

Each arm has a 30-minute, one-V100 allocation. The 28-minute continuation deadline is measured from Python process entry, including import, model initialization, and search; it stops search at the first of cumulative paid search requests 60,000, total outer index 100, or that deadline. The remaining two minutes are reserved for checkpoint flush, summary, and up to three independent fresh checks. The scheduler allocation remains 30 minutes; an incomplete trajectory stays censored and is never retried or extended automatically. Fresh calls are limited to three additional checks per arm, charged separately from search. Parent search and fresh costs remain separately recorded and are included in cumulative totals; they are never replayed or recounted as child calls.

No request is made by preflight. A request is charged immediately before entering the Calculator, including a failed Calculator evaluation. A wall or request-cap denial is recorded separately and does not count as a Calculator request. Checkpoint callbacks persist the exact state after every completed outer attempt and record every landing, including MC-rejected landings, its acceptance decision, LS response/update/prequench data, climb events, request costs, and cumulative index.

## Outcome and qualification

Compare the paired arms at common cumulative paid-request prefixes 5,000, 10,000, 20,000, 40,000, and 60,000 where reached, and at the common completed outer horizon 100 only when all paired arms reach it. `cumulative-readout.json` recomputes the original target gates over parent plus child records and preserves each observation's cumulative cost. The frozen parent's fresh-screen candidate rule is retained: first converged 1.8 Å cage within 0.01 eV of the reference, else lowest-energy new cage. Separately report the stricter Ih identity test (cage and graph-isomorphic to the frozen Ih reference at 1.64, 1.70, and 1.80 Å), force qualification, and fresh E/F result. Also report connected graph classes, fragmentation, LS prequench cost, MC acceptance, failed/censored work, actual Calculator calls, and measured elapsed time. `physical_cage_review` remains a manual review requirement; the harness does not claim to perform that review. A force certificate or graph alone is not physical qualification.

The child summary keeps parent paid requests, child paid requests, cumulative lineage requests, parent fresh calls, and child fresh calls distinct. Once all four arms have summaries, the final arm writes `paired-cumulative-readout.json` with common-prefix metrics linked to the per-arm full observation curves. The original parent readout remains the record of the first segment. No result from this one source and two repeated seeds establishes an independent success rate, a general LS advantage, or equivalence to the paper's PBE surface.

Before the campaign, the root may run the separately tagged one-outer resume qualification on one paired seed. It uses the same checkpoint/config/RNG restoration, runs one actual outer SSW or LS attempt, charges at most 2,000 additional search requests, and has an 8-minute one-V100 allocation with a 6-minute process-entry search deadline. Its output root must be separate from campaign output. It is implementation qualification only and cannot be merged into the 100-step comparison. The root submits the two arms sequentially (`%1`).

## Commands

From the repository root, run the no-PES audit:

```bash
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/c60-source3-continuation-20261012/preflight.py
```

The terminal checkpoint/callback contract can be checked without constructing a Calculator or making a PES call:

```bash
/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/c60-source3-continuation-20261012/preflight.py --terminal-contract-check
```

After reviewing that report, each array task runs one arm into a new, non-existing output root:

```bash
sbatch research/ga_ssw/evidence/c60-source3-continuation-20261012/gpu.sbatch /absolute/path/to/new-output-root
```

Optional one-outer resumed qualification for seed 26100791 only, into a different new output root:

```bash
sbatch research/ga_ssw/evidence/c60-source3-continuation-20261012/qualification.sbatch /absolute/path/to/new-qualification-root
```

Array mapping is fixed in `runner.py`; the four tasks are never reordered or replaced. The read-only preflight and the runner both reject parent checkpoint, ledger, plan, source, or model inconsistencies.
