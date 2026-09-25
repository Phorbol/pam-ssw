# LJ38 early-stage capture probe

## Decision question

If the compact full-budget LJ38 replays remain unsuccessful, determine whether the archived paper-direction arm shows atom peeling in its first three outer attempts while the global-direction arm remains intact, or whether a defect appears only after the biased quench/landing stages. This is a short, development-only diagnostic to locate a stage; it does not tune parameters, establish an unbiased efficiency comparison, or replace the completed compact replay.

The motivating saved observations are outcome-selected: paper seed 25092501 first shows a fragment at outer step 0, paper seed 25092502 at step 2, while the global-direction runs first show fragments at steps 10 and 24. Three outer attempts therefore cover both archived early paper observations and a global early-stage control. They do not measure general rates.

## Frozen protocol

- Arms: existing global and paper initial-direction settings, each with seeds 25092501 and 25092502.
- Initialization and RNG: reuse `run_lj_pilot.uniform_volume_cluster`, its spawned search RNG, `make_settings`, `BoundedSurface`, ledger serializer, and `FullPairLJ`. Following `run_lj38_compact.py`, preflight loads each archived arm's generator and checks exact current-versus-archived generated positions, child search RNG state, and effective settings; it does not compare rounded extxyz coordinates.
- Work: one `run_ssw(..., steps=3)` invocation per arm/seed, so the driver's ordinary initial quench and RNG stream precede exactly three requested outer attempts. There is no progress callback, checkpoint callback, or checkpoint output because the biased-stage adapter cannot be combined with those observers.
- Capture adapter: call the same `quench(displaced, surface, fmax, steps, terms, optimizer, frame, lbfgs_memory)` used by the driver, save its biased endpoint under explicit outer/Gaussian indices, and return `BiasStageQuenchOutcome(relaxed, stage_stopped=False, release_all=False, diagnostics={})`. Writing coordinates and scalar ledgers adds no energy/force requests. End-of-stage geometry diagnostics are deferred to later analysis; any true quench tests on saved biased endpoints are a separate task.
- Evidence: save the initial minimum, each outer start, each available true landing, each biased endpoint, and compact scalar/vector stage records (center, direction, recovered rotation trace/cost, stage requests, width, height in eV, quench requests, true energy, rotation residual, and statuses). Each trajectory summary records its effective SSW and rotation settings. Preserve partial/error runs and paid requests.

## Bounds and stop conditions

Each trajectory allows at most 6000 counted surface requests, including the initial quench and the three outer attempts. The four-run campaign ceiling is 24000 requests and 660 seconds wall time. The runner performs no automatic retry, restart, or extension. A cap or failure may end an arm before three attempts; it is recorded as censored/failed, never silently repeated. Do not launch this runner unless the compact full-budget result leaves the stated stage question unresolved.

## Interpretation gate

Before interpreting this replay, compare each run's first three scalar records with the corresponding archived `outer-steps.jsonl` entries. Any difference in common-prefix status, acceptance, requests, or energies blocks attribution until the replay discrepancy is understood. Matching prefixes support only these four reused seeds and this three-step window. Saved biased endpoints identify what the biased stage produced; by themselves they do not show whether the subsequent true landing preserves or removes a fragment. That requires separately authorized true-quench checks and is not implemented here. No conclusion about full funnel connectivity, general search success, or unbiased efficiency follows.

## Invocation boundary

`python prepare_lj38_stage_probe.py --preflight` performs only path, settings, generated-position, and child-RNG checks and reports zero PES requests. `--execute` first runs the same preflight before creating outputs, then is the explicit PES opt-in. This task prepared the runner and plan only; it did not execute the probe or submit a job.

## Execution decision

Parent reviewed the script and reran zero-PES preflight: all four generated initial arrays/child RNG streams and numerical settings match their frozen sources. After the compact panel completed at4x800000 requests without a target hit, CPU1493151 was submitted on sjtu-caoxiaoming/CPU-MISC/rush-cpu,1CPU,12min ceiling, frozen runner commit947dc50. The same job runs the zero-PES prefix/connectivity analyzer afterward; no GPU, no automatic continuation. Height metadata uses eV (Gaussian weight), not force units.

## Focused recovery after output serialization defect

CPU1493151 used8900 actual search requests in85s. All four runs finished search and saved initial/outer/biased geometries, but scalar summary construction then raised `AttributeError: Atoms has no energy` because SSWResult.best is Atoms, unlike checkpoint.best. No scientific interpretation passes the prefix gate; original failed summaries and analysis are preserved. The in-memory rotation/scalar records were not serialized and cannot be reconstructed completely from coordinate files.

One explicit child replay corrects only the report expression to the already accumulated best_energy. It writes a distinct `stage-probe-repaired-runs` directory. Remaining budget is bounded by3750 requests per arm (15000 total),540s internal and10min Slurm; combined with failed85s this stays below the original12CPU-minute allocation ceiling, and combined maximum23900 search stays below24000. Shorter caps affect only censoring and do not change successful three-step prefixes. No automatic retry, parameter change or further extension. Root will verify all12 scalar steps before interpretation. The analyzer now takes an explicit run directory and writes its derived analysis there, preserving the first failure.
