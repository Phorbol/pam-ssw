# Research three-arm VC comparison runner

2026-09-10. `research/ga_ssw/compare_vc_arms.py` is a Cu-EMT-only research harness. It changes no core optimizer or walker. Its short validation is a wiring check, not a comparison of scientific accuracy or efficiency.

Each arm independently runs the same initial joint true-surface quench, using identical input and VC configuration, and pays its entire cost. Fixed keeps that common relaxed cell throughout and compares E. Posterior-cell-quench (PQC) takes a fixed-cell SSW proposal at its current cell, then performs a true joint cell quench before external Metropolis selection on E+pV. Joint-VC uses the joint atomic/strain kernel and E+pV. These arms have **different accessible configuration spaces**; identical computational budgets do not remove that physical distinction.

One `CappedEMT` owns all E/F/stress requests of an arm. The fixed EF view shares this counter, including initializations, dimer endpoints, failed trial evaluations and final certificates. It returns EF to the fixed kernel while the common underlying oracle obtains E/F/stress. A cap-rejected request is not executed or charged beyond the cap. Each attempted backend failure is retained in the ledger. All stage counters are differences of this one counter.

Each step calls a one-step kernel and extracts its explicit valid **landing**, regardless of internal MC acceptance; returned `current` is never used as a surrogate for the proposal. A separate external MC stream decides acceptance after all required cell quenching. Per-step kernel RNG streams are independently spawned, so unused internal MC draws cannot perturb later proposal streams. These one-step calls include repeated kernel initial quenches; all are counted. The harness is therefore not claimed to reproduce a single native or Python multi-step trajectory bit for bit.

Every valid final candidate, including external-MC-rejected candidates, is saved with ASE structure, objective and force/stress certificate. The certificate is read from an exact structure-keyed E/F/stress evaluation already made by the kernel, without extra requests. Fixed requires atomic force stationarity and records stress without imposing a stress threshold; PQC/joint require both. Intermediate PQC fixed-cell landings, kernel records, failures, stage costs and the complete evaluation ledger remain in the arm output. A denied budget/deadline produces `censored` and preserves last accepted state. No tolerance is relaxed, no failed landing is substituted, and no residual budget is silently borrowed from another arm.

The optional absolute monotonic deadline is checked before each backend evaluation, with a dedicated budget exception handled by the runner. This saves censored progress without external process killing. One in-flight calculator evaluation or file serialization can finish after the wall boundary; this is a cooperative request-boundary limit, not preemption inside native code.

## Local wiring validation

`evidence/vc-arms-wiring/`: same Cu4 fcc input, seed7, at most2 steps and300 requests per arm. All three completed two valid landings: fixed38, PQC42, joint128 calls, **208 total**. The intentionally short wiring preset uses bias0.5 and one Gaussian. It is not the prospective scientific setting. Tests additionally verify cap/deadline censoring, correct stage totals and use of an explicitly rejected kernel landing followed by PQC. Two runner tests passed. These checks remain below the900-request local authorization.

## Prepared, not launched: prospective Cu4 campaign

Reviewable files: `research/ga_ssw/prospective/vc-cu4-three-arm/manifest.json`, `launch.py`, and `source/` containing the full PAM Python package and exact runner snapshot. The launch script uses that source snapshot. No prospective calculation was launched.

- Seeds:29,43,71,101,137,173,211,257; three arms each.
- Each arm:10,000 E/F/stress requests; max10,000 steps so request budget is the usual limiting condition.
- Total cap:240,000 requests; one CPU thread, no GPU or Slurm; shared3,600-second cooperative wall deadline.
- Common temperature300 K, pressure0, fmax0.01 eV/Å, stress tolerance0.001 eV/Å³.
- Developed settings: rotation bias100,14 Gaussians,100 dimer endpoint budget,300 relaxation steps, Safe-total. Joint strain_length3.6 Å is explicitly a development value, not an optimized/unbiased test-set choice.

The short208-call probe had coarse outer-process runtime≈0.86s; naive linear scaling suggests≈992s for240,000 calls. This is only a calibration: the14-Gaussian, bias100 prospective setting and later deformed cells can change costs substantially. The3,600-second wall limit controls runtime separately; this is not a performance prediction.

After explicit user approval, the concrete launch command is:

```sh
python research/ga_ssw/prospective/vc-cu4-three-arm/launch.py --execute --output /path/to/new-campaign-output
```

Without `--execute`, the launcher exits before importing calculators or running work. On deadline, not-started arms are recorded as censored rather than silently disappearing. Longer production/resource use remains approval-gated.
