> 历史快照：其中“缺失/尚未接入”的判断可能已被后续实现取代；当前队列与覆盖状态以 [2026-09-12主线重评](2026-09-12-mainline-reassessment.md) 和当前代码为准。

# Fixed cell SSW / LS kernel contract

Scope: the current `pamssw.standalone.paper_reference.run_ssw` path and its
`run_ls_ssw` / `run_native_ls_ssw` wrappers. This is an evidence inventory for
the fixed cell SSW -> LS -> GA mainline, not a claim of release parity. No
LASP/Java executable is required by this implementation.

| Module / live entry | Current contract and evidence | Supported domain | Known gap or boundary |
|---|---|---|---|
| Initial true quench and outer state (`paper_reference.py:158-248`) | Initial true `quench`; failure raises `InitialQuenchError`. `current`, `best`, minima and request ledger are retained. | Unconstrained atoms; nonperiodic clusters or full 3D fixed cell. | No serialized outer checkpoint containing RNG, LS state, Gaussian history, optimizer history and MC state. |
| Direction sampling (`:132-155`, `:270`) | `paper`: mass weighted random global vector plus a `d>3 Å` pair term with uniform `lambda in [0.1,1.5]`; `global`: only normalized mass weighted vector; `isotropic`: normalized Cartesian normal vector without mass scaling. Sampling uses the working geometry after any LS prequench. The 2013 paper contract records this as eqs 1–2 (`docs/research/ssw2013-paper-contract.md:7-18`). | `paper` for nonperiodic inputs; periodic path requires `global` or `isotropic`, and `translation_only` (`:203-211`). | Native `get_random_mode0/gen_randommode` has different local coefficient machinery; exact generator parity is unknown. The LS paper explicitly orders soft prequench before mode generation; this is not full native generator parity. |
| Rotation solver (`:295-321`) | `ritz` uses `paper_biased_direction`; `dimer` uses `paper_dimer_direction`. Both use finite force differences and bounded numerical residual checks. | Fixed Cartesian atomic coordinates; periodic direction projection is translation-only. | This is a mathematical/numerical substitute for native persistent Broyden/CBD, not instruction parity. Native stopping norm, angular control, history and retries differ or remain unknown. |
| Rotation-only bias (`:282-293`, `:313-316`) | The rank-one `-a((R-R0)·N0)^2/2` term is included only in direction refinement. The 2013 equations 3–6 and paper contract support this separation. | Same fixed-cell domains as direction solver. | It is not a deposited Gaussian and is not the full PAM trust/controller bias. Exact native force composition at every callback remains bounded by existing static audits. |
| LS potential and prequench (`:227-240`, `:257-269`; `ls_cycle.py:55-97`) | Paper LS freezes explicit pair strengths at the true initial minimum, soft-quenchs `E+V_LS`, measures true `E_before/E_after`, then carries frozen LS through the walk. Native-derived settings use recovered table arithmetic/controller through `NativeLSRuntime`. | Paper LS supports full 3D periodic image pairs; native-derived path uses recovered MIC/all-mobile implementation and declares limited parity. | Native periodic save/restore caller, fixed-atom input mapping and complete optimizer exit eligibility are not recovered. No invented bond table/default is supplied. |
| Gaussian accumulation and height (`:322-417`) | Each stage deposits `ProjectedGaussian(center, direction, width, weight)`, accumulates prior terms, then biased-quenchs. Default width is `config.width`; default BP-CBD forward projection targets `0.1 eV/Å`. Optional `ConservativeNativeHeightPolicy` / `MinimalAngleHeightPolicy` computes explicit history-aware height and records preparation. | Fixed-cell atomic walk; height policies are rejected for Eckart section. LS wrappers explicitly forward these policies. | `run_ssw(..., gaussian_policy=PAMCurvatureGaussian(...))` now uses the existing policy's chosen width and weight, actual staged rotation coefficient/anchor, and Gaussian-only history; LS/GA forward it. It is mutually exclusive with height_policy. This does not import the older PAM walker's trust feedback or full proposal machinery. Neither this policy nor isotropic sampling is promoted to a default. |
| Gaussian stop and stage budget (`:300-420`) | Stops at `max_gaussians`, or when measured true energy after the biased walk is below `current_energy`. Optional `bias_stage_steps` treats finite Safe-total `maxiter` as an explicit `stage_budget`, while the default uses `relax_steps` and requires convergence. | Fixed-cell SSW/LS. | `bias_stage_steps` is a Python numerical budget policy. Native whole-release local/outer stop and release schedule are not reproduced; a cap is not scientific convergence. |
| Whole-walk release and true landing (`:439-457`) | After Gaussian stages, optional reconnection is applied only to a copy immediately before the unbiased true quench. All Gaussian/LS terms are removed for the landing; failed landing is recorded and excluded from minima. | Nonperiodic reconnection only; true fixed-cell quench otherwise. | Reconnection restores one audited geometry boundary only; native vapor stop, globalcompress, Ratio_Local and complete post-Allopt chain are absent. |
| MC and cross-step state (`:458-487`) | Landing is archived if true-force converged, then ordinary Metropolis compares true landing energy with `current`; accepted landing becomes next current. LS response update uses true prequench response and selected current structure; native-derived policy explicitly counts completed attempts including MC rejects and failed climbs after converged soft prequench (`ls_native_reference.py:1-7`). | Fixed-cell unconstrained SSW/LS. | Minima are not deduplicated or Hessian-certified in this entry. Native caller association, exact step numbering and numerical-failure branches remain unknown; do not infer extra update gates. |

## Difference from the older PAM `pamssw/walker.py`

The old public `pamssw.runner.run_ssw/run_ls_ssw` constructs `SurfaceWalker`
(`runner.py:8-20`). That walker is a broader adaptive workflow: its
`SoftModeOracle` accepts candidate pools, momentum/anchor candidates, continuity
and history scores, block Krylov selection, energy-bounded anchors and direction
probes (`walker.py:1242-1311`, `:1591-1684`). Its `SurfaceWalker` also owns trust,
step-target and bias-strength controllers (`walker.py:2275-2335`). Those are real
implementation differences, not evidence that every adaptive heuristic belongs
in the fixed paper reference path. The fixed path intentionally keeps one
sampled anchor, one solver choice, explicit Gaussian history and a conventional
MC lifecycle.

The old walker has a richer state/result and relaxation accounting contract,
including post-relax geometry validation and fallback handling
(`walker.py:2694-2730`), whereas the standalone path exposes ASE
`QuenchResult` records and its own explicit request ledger. Results from the two
APIs must therefore not be pooled as if they were trajectory-identical.

## Checkpoint status

Checkpoint capability exists for the reusable periodic atomic substage
`atomic_climb`: `AtomicClimbCheckpoint` stores a completed Gaussian boundary,
anchor, terms, configuration and prior request count, and
`resume_atomic_climb` redoes pending Gaussian work without resampling
(`atomic_climb.py:16-44`, `:72-105`). It does not perform initial/final quench or
MC and does not save optimizer history (`README.md:244-247`). The public
`run_ssw` / `run_ls_ssw` outer drivers have no equivalent complete serialized
checkpoint. Thus checkpoint support must not be advertised as resumable full
SSW/LS trajectory identity.

## Decision boundary

The current implementation is a usable independent fixed-cell reference and
mathematical alternative for the documented domains. It is not complete native
CBD/Broyden or LASP release parity. Unknown native behavior remains marked
unknown; no additional heuristic or old-walker controller should be integrated solely because it exists in another module. The explicit PAM Gaussian adapter has now been compared without changing defaults; it did not show a consistent benefit in the bounded cross-system tests.

## Explicit staged rotation update (2026-09-12)

`SSWConfig(rotation_bias=None, pre_rotation_hvp=k)` now selects the experimental
unbiased dimer presweep followed by the configured dimer/Ritz main solver.
The main anchor is the evaluated presweep direction; its rank-one coefficient
is max(presweep curvature, 0), including frozen LS if present. Both center
requests count against the original 1+rotation_hvp request ceiling. Invalid
combined budgets fail at configuration construction, before any PES request.
Fixed positive rotation_bias retains its previous path. The native caller's
curv>-1e-6 rule and persistent Broyden history are not reproduced by this
independent numerical variant. Full source/probe evidence is in
native-rotation-bias-field-trace.md, and scientific limits in the progress log.
