# Fe7C3-80: user-specified inner/outer tolerance factorial

2026-09-12. The requested tolerance range substantially reduces numerical work in this diagnostic, but does not yet improve low-energy discovery. Eight arms ran on one V100, job1276906, COMPLETED/0:0, 382 seconds. Total 11831 charged E/F/stress requests, including 23 independent fresh checks; two denied requests in the single censored arm. Registered Fe campaign cumulative: 111080. All eight ledger audits have zero consistency errors.

## Controlled question and implementation

Qualified periodic Fe7C3-80, MACE-OMAT-0-small float64, same seeds7/101 and input. Fixed PAM curvature height_width core (not complete PAM feedback), Safe-total history10, maxiter300, 14 Gaussian stages, cell schedule unchanged, two requested outer proposals per arm. Inner .05/.2 and outer .01/.05 eV/Angstrom are user-specified endpoints, not fitted optima. Stress tolerance remains .0001 eV/Angstrom^3. Total budget2000/arm includes three reserved fresh calls; no stage-cap early release.

Added optional BlockSSWConfig.partial_atom_fmax; None preserves prior behavior. The experiment sets .001 to retain the old cell-interleave partial atomic tolerance while outer full-quench force stopping changes. Initial input already meets all requested thresholds. cell_relax uses max(force/fmax, stress/stress_tol)<=1; no hidden joint gradient_tol overrides this. Frozen sources/configuration/input and manifest are in research/ga_ssw/fe7c3-user-tolerance-grid. 18 relevant implementation/interface tests passed, including real Cu/EMT execution.

## Measured outcomes

A completed combined proposal means outer record1 has a valid landing with independently recomputed force/stress meeting that arm's criteria. It is not MC acceptance, a Hessian certificate, or an independently established new phase.

| Inner fmax | Outer fmax | Combined certified | Total requests seed7 / seed101 | Combined delta E seed7 / seed101 (eV) |
|---|---|---|---|---|
| .05 | .01 | 2/2 | 1348 / 2000 | +15.539 / +19.399 |
| .05 | .05 | 1/2 | 1352 / 1999 | +15.539 / censored |
| .2 | .01 | 2/2 | 920 / 1651 | +16.072 / +16.506 |
| .2 | .05 | 2/2 | 877 / 1684 | +14.378 / +17.145 |

All eight arms completed14 Gaussians. Seven combined endpoints and all eight cell-only endpoints passed their own fresh force/stress certificates. All fifteen noninitial landings were uphill and MC-rejected; best energy did not improve. None of the seven combined endpoints passes the former .001 force criterion. Four meet .01, three meet .05 only; all meet the unchanged stress criterion. Do not relabel them as old-threshold minima.

At outer .01, inner .05→.2 reduced total cost by31.75% (seed7) and17.45% (seed101). Completed biased-quench requests fell727→310 and872→502. These are paired pilot observations, not general speedup estimates. Outer .05 did not consistently reduce cost. The .05/.05 seed101 failure is request-limit exhaustion at1997 search calls, not intrinsic maxiter300 failure; both the optimizer and attempted certificate met the same global cap.

## Numerical reproducibility boundary

Nominally identical cell-only prefixes are not bitwise identical across independent GPU processes: seed101 geometry differs by >1e-6 Angstrom by request119 and >.001 by206 when comparing the same outer threshold with different (not-yet-used) inner thresholds. The seed101 cell-only prefix ends at request214/217, so that divergence precedes use of the inner tolerance. Seed7 first differs by >1e-10 at request105; its larger differences at400/418 occur after the cell-only proposal and cannot independently establish unchanged-prefix divergence. Raw evidence is repeat-prefix-diagnostic.json. This shows trajectory sensitivity to small numerical differences; it does not identify the responsible CUDA operation. Consequently individual basin differences cannot be attributed exclusively to the changed stopping threshold. Aggregate budget observations and stage costs remain measured facts, but no deterministic paired-path or causal basin ranking claim is supported.

## Decision

Retain the independent tolerance controls. For the next diagnostic use inner .2 / outer .01 as an explicitly provisional operating point: it completes both seeds and retains the tighter user-requested endpoint criterion. Do not change library-wide defaults based on this reused two-seed Fe pilot. Stop spending the main effort on stricter bias convergence or broader tolerance sweeps. Next priority is native/PAM proposal direction, bias progression and release/reference-state semantics, using saved exact stage inputs to isolate numerical trajectory divergence before another efficiency comparison. The remaining scientific issue is uphill proposal quality, not merely numerical stage completion. Independent systems/seeds and adequate final structural checks remain required before general performance claims.

## Artifacts

- research/ga_ssw/fe7c3-user-tolerance-grid/tolerance-summary.json: counts, failures and cross-threshold certificates.
- research/ga_ssw/fe7c3-user-tolerance-grid/stage-cost-summary.json: completed bias-stage costs.
- Full all-pairs structural matching exceeded the tool execution window and produced no comparison-summary.json. No cross-arm mutual-distinctness claim is made. Focused initial-versus-landing checks are handled separately.
- research/ga_ssw/summarize_tolerance_grid.py: zero-PES ledger/certificate auditor; materialized plan-used configuration takes priority.

Focused identity check completed (exit0, zero PES): all15 certified noninitial landings differ from their initial structure under strict/default/loose pymatgen tolerances (0/45 same-as-initial). This includes seven combined endpoints. Artifact: research/ga_ssw/fe7c3-user-tolerance-grid/focused-identity-summary.json. Cross-arm mutual distinctness and Hessian stability remain unverified.
