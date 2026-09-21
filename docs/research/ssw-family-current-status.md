# Independent SSW family: implementation and evidence

**2026-09-11 core update:** Explicit joint atomic/cell LS preparation is now
available through `run_vc_ssw(..., ls_prequench="joint")`, with frozen periodic
pairs, physical enthalpy response and fresh softened joint-gradient/force/stress
checks. The fixed-cell preparation default is unchanged. The matched Fe7C3-80
MACE comparison completed all8 requested attempts at5777 EFS versus5502 for
fixed preparation, with0 valid candidates in either group. Removing residual
cell gradients therefore did not establish improved exploration. See
[joint-LS design](joint-ls-preparation-design.md) and
[the complete comparison](fe7c3-80-joint-prequench-comparison.md).
The earlier listed306-test result below is historical; the latest complete
interface/numerical regression is397 passed,1 skipped, recorded in
`research/ga_ssw/evidence/joint-ls-prequench/regression.log`.

**Later 2026-09-11 update:** [XXXII RC-VC](xxxii-replicated-rc-vc-conclusion.md)
now completes 12 biased stages, full-cell true quench and independent fresh
checks on the corrected replicated GAFF representation (3559 API). The higher
energy candidate is MC rejected; two stricter endpoint quenches preserve a
+0.31314 eV difference. Central Ritz is available through an explicit optional
VC/RCVC solver API, with evaluator-call budget separate from the default dimer
HVP budget. Full real-trajectory public-API replay matches all3557 search calls
exactly; this replay adds no physical evidence. Two-step-size full finite-cell
Hessians now have519 positive nontranslation eigenvalues at each endpoint.
The three explicit periodic StructureMatcher tolerances all return no match;
this supports different numerical minima in the supplied model, not a new stable phase. Latest direction-mask regression:347 passed,1 skipped in the user-site reference-compatible
environment; isolated NumPy2.0.2/ASE3.26.0 gives339 passed,1 skipped and2 old Cu13
reference-trajectory failures, reproduced with prechange frozen code. See
`public-direction-api-verification/root-test-summary.json`; tolerances unchanged.

**2026-09-11 update:** see [overall progress](2026-09-11-overall-progress.md).
The MACE-OMAT C60 LS step completes its numerical lifecycle (12 biased stages,
true quench,2 fresh checks) at1,395 E/F/556.698 s, plus4 precheck calls. Its
fresh-matched landing has fmax0.00575745 eV/Angstrom but is C58+C2 separated by
16.26250 Angstrom,8.68519 eV above the MACE initial minimum, and MC rejected.
Numerical landing success is distinct from intact-C60 search success. No V100
or GPU job was used. The detailed inventory below retains its evidence dates.

2026-09-10, research/ga-ssw-behavior-parity. These implementations use ASE
calculators and Python numerical code, with no LASP/Java runtime. This inventory
supersedes older missing-RC/periodic-GA status paragraphs; it does not supersede
original reverse-engineering evidence or turn independent corrections into parity.

| Scope | Implemented lifecycle | Important remaining boundary |
|---|---|---|
| Fixed-cell SSW | Random/soft direction, conservative Gaussian continuation, true quench, MC, observations/costs | Native CBD history is an independent solver substitution; optional conservative native/minimal-angle heights implemented |
| VC-SSW | Joint atomic/log-strain escape, exact force/stress pullback, E+pV quench/MC | Explicit independent coordinate metric; native stored stress producer still under audit |
| Sequential cell/atomic SSW | Cell/atomic stages and joint final certificate | Separate algorithm from joint propagation; eight-run material gate and 12 finite-cell qualification tasks completed; one-seed scope |
| LS-SSW | Paper amplitude response and native-derived cycle/response drivers | No consistent advantage in current C4H6 paired tests; not a promoted default |
| Periodic LS / VC-LS | Frozen periodic image bonds, extensive penalty, exact stress and fixed-cell soft prequench | Conservative independent extension rather than native MIC behavior |
| RC / RC forest | Tree torsions, relative molecule pose, exact pullback, unrestricted true quench | Explicit topology/metric; closed-loop constraints unsupported |
| RC-VC | Rigid interiors, relative pose, joint strain search and unrestricted cell quench | XXXII full lifecycle, fresh checks and finite-cell Hessians qualified in converted GAFF; broad physical effectiveness unvalidated |
| Fixed substrate | Active Cartesian SSW and quench, FixAtoms, partial PBC, reaction-force diagnostics; separate optional direction-fixed atoms | Cu28 EMT lifecycle and exact default trace qualified;514atom MACE full steps and landing Hessians completed; common initial required further descent; no general holonomic or variable-cell substrate manifold |
| GA TYPE0 / TYPE3 | Atomic/whole-molecule/mutable-unit proposals and complete three-stage fixed-cell search | Recovered packing families and rigid-unit auxiliary refinements implemented; unrelaxed seeds are not certified minima |
| GA TYPE1 / TYPE2 | Crystal / whole-molecule periodic proposals, full three-stage VC walks, periodic routing and independent identity | Explicit budgets and corrected native bugs; native schedule/multipliers not reproduced |
| GA TYPE4 | Full support-preserving proposal families, constrained three-stage SSW, separate auxiliary-cost ledger | Explicit caller routing/identity; broad supported-material effectiveness unvalidated |

All minimization certificates are stationarity checks until independently
qualified. RC constraints apply to proposals, not the final physical minimum.
Fixed-substrate certificates instead deliberately retain the substrate constraint.
Routing descriptors do not identify kinetic funnels or establish transition edges.

## Current measured evidence

* Released-module standalone plus material-qualifier checks: 306 passed, 2 skipped,
  20.93 s (`/tmp/ssw-family-round8.log`). Formula/interface regression evidence only;
  this includes RC/VC frozen-objective preservation at failure exits.
* C4H6, GFN2: six paired ordinary/paper-LS/native-LS runs completed, 24,866 E/F
  including 18 fresh checks. No consistent cost/coverage winner. One ordinary
  landing dissociates despite a small force; retained as a physical failure.
  Earlier censored dataset cost 12,000 additional requests and remains separate.
* Butane RC: 66 E/F including fresh endpoints; anti-to-gauche candidate, rejected
  by MC and retained in observations. Bound S22 water-dimer forest: 183 E/F;
  bound endpoints, no established new minimum. The prior manually placed dimer
  dissociated and remains a negative result.
* Cu(111)+Cu EMT constrained SSW: 62 E/F, FCC-to-HCP adsorption registry, energy
  change -0.00525893 eV; fixed support/cell exact. No barrier/Hessian/efficiency claim.
* Artificial rigid Cu4 EMT RC-VC: 37 E/F/stress, exact rigid proposal geometry
  and unconstrained final certificate. Integration evidence, not molecular-crystal
  production validation.
* AlOH26 / brookite48 MACE four-arm gate: all eight runs terminal, 15,467 total
  requests. Seven runs reached the request limit; AlOH block completed both
  attempts. AlOH joint yielded one candidate 0.200585 eV below its prepared start;
  brookite joint yielded no valid landing within its 2,000-request budget. These
  are one-seed feasibility observations, not a ranking of search efficiency.
  Independent qualification completed with 5,708 additional requests / 3,256.17 s
  within the declared CPU cap. All 16 source endpoints pass fresh original
  tolerance checks; all 12 domain-specific strictly refined structures have
  positive projected Hessians at both steps. The refined Al8O14H4 joint landing
  remains 0.2005702 eV below the refined variable-cell start. This is finite-cell
  model-specific evidence, not full phonon or DFT phase qualification.
  The frozen source/config and full failed/censored denominator remain intact.

Evidence folders reside under `research/ga_ssw/evidence/`; the search gate is
`research/ga_ssw/prospective/complex-vc-feasibility/`. Tests or a completed stage
never establish universal gains. The criterion for retaining a source-derived
strategy is reproducible benefit at comparable total cost across the relevant
systems, not resemblance to the original binary.


## Conservative native height comparison

`run_ssw(..., height_policy=ConservativeNativeHeightPolicy(...))` now supports
recovered level-dependent history rewrites and 87-degree force-angle growth.
The profile corrects native old-force double counting and freezes every term
during each optimization. The physical/LS directional secant curvature is
reconstructed by removing the known rotation-only rank-one bias; native LS
curvature semantics remain unclosed. The default forward-force policy is retained.
Prepared objective/history/costs survive quench exceptions. Eckart section geometry
is rejected for this profile until its separate force-angle contract is established.

A frozen paired C4H6/GFN2 and Cu13/EMT comparison was launched with seeds 3/17,
two outer attempts, 6,000 total E/F per arm (including up to three fresh checks),
eight-run denominator and 600 s shared wall ceiling. The ELF-stored values
(2,.2,level1,10,1.05,1.25) are an explicitly source-derived experimental setting,
not asserted universal defaults. Source, input, plan and all outcomes are in
`research/ga_ssw/evidence/conservative-native-height-two-system/`. No retry or
tuning against this dataset is planned. Different returned work at the request
limit is censored and cannot be represented as a completed-work speedup.

## Follow-on Cu13 qualification and interpretation

All six Cu13 height runs (forward/native/minimal-angle, seeds3/17) contributed
all 18 initial/landing structures, including rejected endpoints, to independent
post-search qualification. It used 880 additional EMT E/F, 3.025 s, with no
feedback into the frozen searches. All points converged at 1e-5 eV/Angstrom.
Four sorted-pair-distance groups have positive internal Hessians at both 1e-4
and 5e-5 Angstrom central-difference steps after removing six rigid modes.
Fingerprint groups are distinguishable candidates, not an exhaustive/injective
basin classification. Qualification pertains to refined geometries; maximum
raw-to-refined fingerprint change reaches .03405 Angstrom and is retained.

| Strategy | Observed refined groups | Search + fresh E/F |
|---|---|---:|
| Forward-force baseline | 0,1,2 | 2396 |
| Conservative native profile | 0,2,3 | 3069 |
| Minimal-angle profile | 0,1 | 2029 |

The common initial group0 energy is9.36135788 eV; groups1/2/3 have energies
10.46286188/10.48849784/10.17423741 eV. All new groups were MC-rejected, yet
correctly retained as PES observations. Native-only group3 gives evidence of
complementary local-minimum coverage at higher search cost, not improved global
minimum discovery. Two seeds/two attempts cannot establish a universal scheduler
or optimal strategy. Artifacts: `evidence/height-cu13-qualification/result.json`.


TYPE3 mutable internal libraries and supplied-seed TYPE2/3/4 initializers are
now implemented. These are geometric generators, not physical minimum builders.
The original periodic/molecular/surface initializers themselves rely on supplied
structures; only TYPE0 includes a wider composition-only packing repertoire.
The released-module regression snapshot above includes auxiliary initializers,
all recovered packing families, and corrected TYPE4 exception costs/TYPE3
contiguous topology preflight. Test counts remain implementation evidence only.


## RC numerical domain and initializer completion

A structurally null torsion (an entire child subtree on the shared axis) is
rejected before oracle work. Optimization-only RC forest/RC-VC wrappers now use
the principal root-rotation ball ||omega||<pi; this is the SO(3) injectivity domain,
not a fitted search angle. Arbitrary-angle raw geometry remains available.
An invalid optimizer trial reports a domain failure through the existing solver;
no unimplemented history-preserving rebase or boundary contraction is claimed.
The S22/GFN2 retained-parameter path still completes with 183 E/F; see
`rc-principal-rotation-domain.md` for the counterexample and full cost evidence.

All TYPE0 source packing families now have independent primitives in `packing`
and `initialize_type0_regular`; explicit source geometry/occupancy/radii and finite
attempt budgets remain mandatory. The original empirical density can exhaust a
budget before producing a complete structure; those partial results are retained,
not passed off as valid population members. TYPE1/2/3/4 seed expansion and TYPE3
optional rigid-monomer auxiliary LJ optimization are implemented. Native auxiliary
angular-gradient and terminal-batch omission defects are documented corrections.
This completes the recovered geometric families, not broad production validation
of the resulting population or exact native random streams.


## Native curvature timing recovered

`native-curv-real-provenance.md` closes the height field's immediate writer with
12 original-instruction cases: native curv_real uses (tf0-fa) dot n_old / dr
before the current direction rotation; rotstep=0 sets zero. The Python profile
removes a known rank-one term from its declared direction solver curvature.
The expressions can agree for matching direction, force samples and surface,
but the returned direction may differ from native's incoming direction.
The native CBD callback adds LS before this contraction; it is not a bare-PES
curvature. Fresh pairing of the initial saved coordinate/force on every native
prequench exit is still unclosed. No extra physical
evaluation or compatibility parameter was added from this timing evidence.


## Local-optimizer memory: concrete gain, still experimental

The source-backed non-Ih C60/GFN2 one-step experiment completed ordinary SSW
with 384 E/F and a near-return to the same cage. Paper-LS spent 1,422 E/F but
reached the existing 400-step cap in biased stage8, without a valid landing.
All 423 evaluations of that failed stage were replayed from the original stage
start with **zero coordinate discrepancy and zero new physical calls**. Its
400 accepted curvature pairs were positive; negative-pair contamination is not
the explanation. The final modified fmax was .0113546 versus the .01 target.

| Frozen C60 stage8 solver | Optimization E/F | Accepted steps | Modified fmax | Additional fresh E/F |
|---|---:|---:|---:|---:|
| Safe-total, original memory10 | 423 | 400 | .0113546 (not converged) | included in original evidence |
| Original ELF numerical kernel | 314 | 309 | .0097104 | 1 |
| Safe-total, only memory400 changed | 264 | 248 | .0098579 | 1 |

The memory400 run preserves all other Safe-total choices. Offline prefix replay
shows the first direction difference only when retaining more than ten pairs
first affects the two-loop recursion. This is a causal mechanism on a selected
subproblem, not a universal optimal memory or an end-to-end LS recovery. The
three final modified energies differ; the unconverged memory10 endpoint actually
has the lowest of those energies. Threshold convergence and energy ranking are
separate facts. New ELF and memory400 experiments cost 315 and 265 E/F; the
original 1,806-E/F two-arm C60 experiment is unchanged.

On all 31 prior frozen Cu13 failures, memory400 converged 31/31 using 2,804
optimization +31 fresh E/F, versus memory10's existing 3,058 optimization calls.
Per case, cost was lower/higher/equal in 19/5/7 cases. These are the complete
failure-selected subproblems, not an unbiased whole-search benchmark. An initial
research-runner postprocessing property error was repaired from saved trajectories
without rerunning optimization; repaired-summary.json is authoritative, with
44 seconds within the original 120-second wall budget.

The public memory10 default remains unchanged. The new-seed full-search comparison
has completed: Cu13 EMT ordinary SSW and C4H6 GFN2 paper-LS, seeds29/71,
two outer steps, four paired runs. Total requests decreased from 11,822 to
9,009 (23.8% in this set); all four pairs saved requests. All eight runs and
16 landings completed; 24/24 independent fresh checks passed. Total development
cost was 20,831 E/F and 51.47 seconds. No accepted new-basin gain was established;
two near-return acceptances and 14 MC rejects remain explicit. These are new
seeds on known systems, not unseen-system validation.

An explicit `lbfgs_memory` option is implemented for fixed-cell SSW/LS and is
now propagated through generalized/VC/RC paths. None retains10; only Safe-total
accepts an explicit positive integer. No global monkeypatch is required. Fixed-cell
cached C60 replay reproduces every saved coordinate for default/10/400 with zero
new PES calls. Sources: hard-c60-safe-history-ablation.md,
cu13-safe-history400-regression.md, safe-history-newseed-e2e-results.md,
explicit-lbfgs-memory-design.md.

## Hard-C60 full-step history follow-up

The frozen public-memory400 paper-LS seed3 run finished with 1,016 search plus
1 fresh E/F request in 501.804 s, below its 2,000-request/900-s limits. Nine
biased quenches converged, but Gaussian index9 rotation stopped at residual
0.03911687 eV/Å² above tolerance0.02 after99 HVPs plus1 center force request.
There was no true landing quench, new minimum or MC acceptance. The old memory10
run failed earlier at a biased quench; both full-step attempts remain failures.
Their different failure costs do not establish search efficiency.

The first coordinates diverge at request41; equal stage numbers afterwards are
not identical frozen objectives. Ordered zero-PES replay reproduces the failed
rotation exactly without relaxing its tolerance or enlarging its budget.
Artifacts: `research/ga_ssw/evidence/hard-c60-gfn2-paper-ls-memory400-single-step/`.

## Hard-C60 rotation precision result

At that frozen objective, a valid fresh Ritz run passed in9 requests while the
saved dimer had failed in100. However, independent force evaluation at three
calculator precision levels made both saved final directions pass the same0.02
residual threshold. Their ranking changed; a robust Ritz advantage is unproven.

A paired rerun of the unchanged dimer, fresh calculator per arm and reused
within each arm, then found:

| GFN2 accuracy | force requests | direct residual | converged | wall seconds |
|---|---:|---:|---|---:|
| 0.001 | 100 | 0.0297392733 | no | 50.7905 |
| 0.00001 | 28 | 0.0151592221 | yes | 18.0811 |

This128-request comparison supports addressing force precision before replacing
the rotation algorithm. It is one frozen problem, not a full LS-SSW recovery,
global efficiency claim, or basis for a universal calculator accuracy default.
The combined local campaign costs1,172 E/F: full-step1,017, invalid Ritz9,
valid Ritz9, fixed-direction precision9, paired dimer128. The invalid attempt
and split precision collector remain visible in their original artifacts.
See `hard-c60-mode-accuracy-contract.md` and
`dimer-precision-paired-v1/report.md` under the above evidence directory.

The subsequent whole-run accuracy0.00001 control has also terminated:1,030 E/F,
738.9192 s,10 completed biased quenches followed by an SCF250-cycle failure at
Gaussian index10. It produced no landing; the sole fresh check certifies the
retained initial structure only. Thus the frozen-rotation recovery has not yet
translated into a complete hard-C60 LS step. Including this control, the above
local campaign costs2,202 E/F; all unsuccessful attempts remain in the ledger.
See sibling evidence directory
`hard-c60-gfn2-paper-ls-memory400-accuracy1e5-single-step/report.md`.

Cache-only replay locates the failure after successful rotation, at the displaced
background-force request. All successful replay coordinates match exactly.
Both precision arms already contain separated C58+C2 biased intermediates;
neither has a true landing. Published C60 inputs also include `globalcompress`,
`vapor_cri`, and `Ratio_Local`, whose full action is not reproduced in the current
Python walker. Static caller evidence establishes only a conditional native
vapor-related local-stop path; compression and subsequent MC behavior remain
unclosed. See `c60-native-fragmentation-boundary.md`. This is an explicit
algorithm-completeness gap, not evidence to introduce an arbitrary restraint.
