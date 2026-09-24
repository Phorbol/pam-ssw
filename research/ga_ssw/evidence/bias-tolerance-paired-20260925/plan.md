# Paired inner-bias tolerance probe (prepared; not submitted)

## Question and decision

With full Gaussian climbing depth retained, does changing only the inner
bias-quench force threshold from the inherited `0.1` to the user's proposed
upper bound `0.2 eV/Å` lower its measured E/F cost while still producing a
force-qualified landing different from the post-initial-quench start?

This is a development numerical-sensitivity probe, not parameter tuning,
default selection, or a claim of a generally optimal tolerance. Near a stable
minimum of the biased surface, after removing rigid/constraint null modes and restricting to a locally positive-curvature subspace, the local approximation
`δR ≈ -H_eff^{-1} g` allows soft modes to amplify a residual force. It does not
guarantee that the looser threshold preserves useful escape progress; that is
why each arm retains the complete Gaussian depth and is measured end to end.

## Paired arms and inherited protocol

There are four starting cases, each run at `bias_fmax=0.1` and `0.2`: C4H6
native-LS seeds 61 and 67, and C60 prospective starts/seeds 17093 and 17094.
Each pair reads the same original input and uses the same main RNG seed. Each
run starts afresh, including its own initial true quench and newly initialized
LS state. `run_ssw(steps=1)` means one additional outer attempt. This first
attempt begins with a newly initialized LS controller, so the result describes
that start-up attempt only; the two seed values are repeated randomized probes,
not independent initial structures. Changing the tolerance can change RNG
consumption and later draws, so this is a matched-input/seed comparison, not a
claim of bitwise-identical trajectories.

Everything except `bias_fmax` is loaded from and checked against the saved
effective configuration of the corresponding original runner: MH-1/omol,
width, full NG25 C4H6 or NG12 C60 Gaussian depth, true `fmax=0.03 eV/Å`, safe
L-BFGS total-quench settings, native LS, and recovered-rotation settings.
The C60 native-MC settings are inherited as well. The C4H6 original runner's
default MC behavior is retained. No source under `pamssw/` is changed.

The scientific motivation is limited to the existing evidence: an earlier
C4H6 depth probe found early quenches often returned to the start, so removing
Gaussian stages is not supported; across four selected C60 outer paths, biased
quench costs summed to 2698/3211 requests and made up a large share of their
measured search cost. Those findings motivate testing tolerance while
preserving depth; they do not predict success.
The tested upper value `0.2 eV/Å` comes from the user's stated range, not from
this experiment's outcomes.

The source paper's Step 5–7 describes repeating Gaussian climbing through the
selected depth and then removing LS/Gaussian bias for the real relaxation; its
SI specifies `NG=25`, `ds=0.1 Å`, and `ftol=0.01` for the cited C4H6 setup.
Those sources do not establish the `bias_fmax=0.1/0.2 eV/Å` pair used here.
This therefore probes numerical sensitivity in the independent ASE/Python
implementation and makes no original-code numerical-parity claim.

## Measurements and interpretation

Per arm retain initial true-quench and outer-record status, completed Gaussian
stage count, biased-quench requests by stage, all search E/F requests (including
initial and landing work), and wall time. Independently fresh-evaluate the
post-initial-quench start and any returned landing, at most two fresh E/F
requests per arm. Report fresh true-force qualification at `0.03 eV/Å`,
composition/cell/PBC checks, continuous structure change using the existing
graph-compatible proper-rotation Kabsch RMS, connectivity, and (for C4H6) the
existing CCCC torsion. C4H6 connectivity uses the existing element-specific
cutoff (`HC_BOND_LENGTHS + 0.1 Å`); C60 reports graph metrics at the previously
declared `1.8` and `1.64 Å` cutoffs. Graph equality does not establish the same
basin, and no new RMS threshold is introduced to force a binary
changed/unchanged label. Report fresh force qualification and structural
comparisons as separate fields.

Primary comparison: paired full E/F cost to a fresh-force-qualified landing,
including initial quench and fresh validation, with failures and censored
attempts retained. Also report search and fresh requests separately. Report
graph relations and continuous structural differences without imposing a new
threshold. A lower cost at `0.2` while retaining fresh-qualified landings in
both systems supports a follow-up under a
prespecified multi-step protocol, not promotion to a default. Mixed/null
outcomes retain `0.1` pending a more diagnostic cause; no extra parameter values
or retries are authorized by this protocol.

## Budget and stops

Each of 8 arms has at most 6000 search E/F requests and 2 fresh E/F requests.
Hard ceiling: `8 × (6000 + 2) = 48016` requests total, one V100, 30-minute
Slurm allocation, cooperative deadline at 29 minutes. No automatic retries,
resume, added outer steps, or budget extension. If an arm hits its cap/deadline,
record it as censored; continue only with already-budgeted later arms while
time remains. No submission is part of this preparation task.

## Provenance and files

`run.py` resolves original inputs and saved effective configurations by path,
checks their SHA256 values plus model SHA and unchanged `pamssw/` tree, and
records runtime HEAD, configs, actual requests, and output checks. Raw request
ledgers and result directories are runtime artifacts under this evidence
directory and are never overwritten. The original source records are not
modified. This protocol and runner are prepared artifacts only until separately
reviewed and explicitly launched.

## Preflight and execution decision

CPU1483548 completed1s with no PES: all8 configs constructed from actual public interfaces and paired inputs matched; only bias_fmax differs within each pair. Main-agent reviewed raw request accounting, fixed-tree import, NativeMC field mapping, and removal of unsupported geometry thresholds. Proceed with the existing48016-call/30-minute ceiling, no retry or expansion.

Execution: frozen runner222f079; GPU1483551 submitted on4V100/rush-1o2gpu, group account; CPU1483553 depends afterany for5min structural readout. No explicit sbatch export override.

Scheduler-only amendment before execution:4V100 was fully allocated and the job waited on Priority. Live inspection found8V100V0 UP with4 idle V100-SXM2 nodes and the same permitted rush-1o2gpu QOS. Updated the existing pending1483551 to8V100V0/6 CPUs (partition's per-GPU default); one GPU,30min, single-thread math, same account and48016-call ceiling remain. Allocated billing weight180 matches the original180; this is a scheduler weight, not a currency-price claim. Started2026-09-25 01:39:51 on8v100v0n01. Before/after scheduler records retained. Frozen scientific plan/runner unchanged; this is an explicit resource-location amendment, not a rerun. Pairwise timings remain from the same job/node; no wall-time ranking against old4V100 campaigns.
