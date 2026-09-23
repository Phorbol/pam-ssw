# Same-population SSW versus GA-SSW: fixed development protocol

## Question and decision

After completing the approved state/checkpoint integrations, test whether the
existing GA scheduling/proposals improve physical low-energy discovery at the
same total cost as independent SSW walks from the same population. This is a
whole-strategy comparison, not isolated proof that crossover alone helps.

Competing explanations: nonlocal proposals can access useful structures that
local walks miss; alternatively their quench/failed-candidate cost can outweigh
their benefit. More observations alone discriminate neither explanation.

No core algorithm change, new descriptor feature, LS activation, or VC/RC work.
No parameter selection based on these results. Complete current mainline source
is used in both arms; numerical kernel configurations are identical per system.

## Inputs and model

Four existing raw random60C structures (17093–17096), same prior ball/rejection
sampling protocol, nonperiodic, no known cage seeded. All four are now reused
**development** inputs; 17095/96 were held out in the prior rotation comparison,
not held out for this new study. C60 uses MH-1/omol float64 CUDA. The independently
qualified MH1 reference energy is -62215.39337032347 eV; its structure is
`c60-mh1-qualification-20260919/reference/fresh-final.traj`. The older OMAT
reference geometry is not mistaken for an MH1 energy certificate.

Cu13 uses the three existing force/Hessian-qualified EMT minima from
`atomic-cu13-ga/initial.extxyz`; this is a different PES class and an inexpensive
integration/coverage comparison, not a difficult random-cluster benchmark.
Both methods receive identical ordered inputs within each system.

## Budget and allocation

Search RNG seeds 3 and17 per system. C60 60000 E/F requests per arm; Cu13 20000.
Four arms per system: total maximum 240000 C60 +80000 EMT search requests;
at most4 post-search fresh checks per arm. Raw request counts and wall time
are reported; they are not asserted equal to internal model-forward counts.

Both arms explicitly quench all raw starts by the same current GA initial-quench path
(Safe-total for the chosen configs, with identical history setting; BFGS only
for the fallback branch used by other optimizer selections)
with common force/step limits. The baseline then divides its remaining budget
equally over the qualified starts in original order, carrying unused allocation
forward. Each start runs an independent SSW trajectory, with no best-seed restart.
The GA arm uses the same total cap, including all offspring failures and quenches.

GA quick10 and generation-short10, fine10000 outer-step ceilings; one GA
generation, eight requested candidates, one region/fine region. Paper quick
10–100 and fine100–10000 ranges motivate stage lengths; one generation/eight
candidates/one region are explicitly a bounded study design inherited from the
existing Cu13 controller protocol, not optimal or C60-native settings. Returned
candidate count may exceed8 because the reconstructed proposal returns batches.
Reaching the E/F cap can truncate any phase, and failure to reach GA or fine
must be reported, not repaired by reallocating after seeing the result.

C60 retains the previously frozen ordinary Broyden/Safe-total500/NativeMC
configuration, fmax.03, bias_fmax.1, fd_step.001. No simultaneous recovered-CBD,
full-direction or LS comparison. Cu13 retains its own existing SSW configuration.

## Descriptor provenance and intentional limits

C60 NNA C-C table=2*1.278556=2.557112 Angstrom from released ElementPara and
BasicInfo; NeiSize1.5, six weights .3,.2,.2,.1,.1,.1, similarity.001 and global
energy window10eV from shipped TYPE0 settings. These are source-backed values,
not C60-validated chemical bond cutoffs or canonical paper DCCD. The source
TYPE0.getGA has no getLimit filter; retain empty proposal_bond_limits rather
than applying the configure-file comment for another call path.

Freeze first3 **raw input** descriptors in supplied order as projection anchors;
never include the cage target. This is an explicit operational NNA basis, not
paper-style orthogonal minima anchors. Full-fingerprint row ordering avoids the
known atom-numbering defect. Cu13 retains its archived descriptor data.

Sources: `decompiled/sgn/nna/BasicInfo.java:76-93`,
`decompiled/sgn/other/ElementPara.java:243-251`, `ga_Interface/TYPE0.java:24-68`,
`GA-SSW_examples_run/global_exploration/input-templates/TYPE0-LJ75/configure.non`,
`literature/GA-SSW-user.txt:228-252` under the original research archive.

## Acceptance and next decision

C60: report target cage and reference-energy success separately; geometry is
connected 60C, 3-regular/3-connected planar graph with12 pentagons/20 hexagons,
primary cutoff1.8Angstrom, sensitivity1.64/1.7. Energy criterion reference+.01eV,
and independently force-qualified best/cage endpoints. Keep disconnected or
noncage low energies visible; graph criterion alone is not a PES certificate.
Cu13: independent force/composition/boundary checks, best qualified energy and
geometric diversity with stated comparison limitations. No DFT claim.

Preserve failures, partial phases, budget/wall censoring, initialization costs,
and all available rejected valid landings. Compare matched-seed endpoints and
cost boundaries; do not infer smooth discovery curves if the observation time
is only known to a phase boundary. Two seeds do not estimate reliable success
probabilities. Mixed/no benefit ends this fixed comparison without tuning.

Before expensive runs: CPU runner/counter tests and input/descriptor preflight;
then bounded GPU execution on at most2 V100 jobs concurrently, each100min.
CPU measurements run on CPU nodes. No automatic resubmission or budget expansion.
Inputs/plans: `research/ga_ssw/evidence/population-comparison-20260923/`.

## Pre-execution review correction

The initial preparation note called the GA initial optimizer BFGS based on an
older audit. Live `paper_ga.py:495-502` already selects configured Safe-total,
SciPy or ASE line search, forwarding history, and uses BFGS only otherwise.
The study baseline mirrors that current branch before any scientific execution.
CPU1463275 zero-PES input/descriptor preflight passed:4/4 C60 and3/3 Cu13 raw
projections distinguishable; MH1 reference passes the cage graph. This does
not certify post-quench parent diversity or predictor accuracy.

## Driver verification and launch

Runner is frozen in the separate clean `research/ga-matched-search` worktree at
`ec01115`; actual run provenance records the full SHA. Independent review found
and fixed per-start quota termination, initial optimizer routing, duplicate
fresh checks and wall-censor reporting before formal execution. CPU1463423
passed16 focused checks; after the final wall-boundary regression CPU1463485
passed17. No core algorithm was changed. Wall guard refuses new PES requests
and GA stops at its next existing phase boundary; scheduler ceiling is100min.

CPU1463493–1463496 are the four prospective Cu13 arms. No result is claimed
until result files, per-request ledgers and independent endpoints are checked.
TYPE0 nonperiodic crossover may replace50Angstrom bookkeeping cell with zero
cell; PBC remainsfalse. The output records this distinction; an exact cell flag
is not by itself an isolated-cluster physical boundary failure.

## Cu13 result and C60 release

CPU audit1463551 verifies four consecutive request ledgers, each exactly20000;
all stage sums close and no paid PES request failed. Baseline uses all3 starts
with6666/6666/6665 requests after3 initialization calls. All four best endpoints
remain the same initial minimum9.361357881570044eV, independently reproduced
with max force9.9433e-6eV/Angstrom. This study shows no best-energy gain on these
inputs; it does not demonstrate absence of a GA benefit on harder starts.

GA seed3 spends13314 on quick exploration,412 on13 offspring quenches,6036
on generation-short,235 on fine, plus3 initial calls. Seed17 spends16557 on
quick,368 on13 offspring quenches,3072 on generation-short, plus3 initial;
it never reaches fine. This is a frozen-schedule/budget limitation, not a
software failure or reason to retune the completed study. Search requests
are distinct from calculator cache misses; both counts remain in summaries.

First offline readout incorrectly rejected surface=true because ASE extxyz
parses that literal as boolean True. Corrected only the analyzer, reran the
same raw artifacts on CPU1463551; the original diagnostic log1463533 remains.
No search was rerun or changed. Readout: analysis-cu13.json under the evidence
root; raw run directories retain all inputs, ledgers, checkpoints and structures.

C60 runs released with frozen sourceec01115:1463566/1463568 (seed3 SSW/GA),
1463567/1463569 (seed17 SSW/GA, each dependent on the corresponding first arm).
At most2 V100 concurrently,60000 requests and90min software wall per arm,
100min scheduler ceiling; no result yet. See gpu-submission.json.

## C60 configuration correction before search comparison

First launch failed due to a research-plan field typo: NativeMCSettings accepts
energy_tol, not energy_tol_eV. GA1463568 failed before any PES request; cancelled
SSW1463566 had261 paid initialization records (an in-flight request may be
unlogged),1463567 never ran,1463569 was cancelled during setup. Preserve the
original folders, logs and failed-launch-accounting.json outside scientific
comparison. This setup failure is not evidence about SSW or GA performance.

Only the field spelling changes in c60-seed3-v2.json/c60-seed17-v2.json, still
0.1eV. CPU1463615 validates both revised C60 and both Cu13 complete config
objects, walker options and raw descriptors, zero PES with backend loading
excluded. Numerical source remains ec01115, no algorithm/default changes.
Replacement runs1463630/1463632 (seed3),1463631/1463633 (seed17 dependent)
use new v2 output folders. Scientific budget remains60000 per arm; failed
launch cost is additional development overhead, explicitly retained.

Independent read-only review found no further constructor mismatch. GA uses
one sequential RNG stream whereas baseline walks each use seed+input-index;
matching seed labels does not couple per-move randomness. This is a comparison
of complete search strategies, not an isolated causal estimate for crossover.

Follow-up harness fix (not used to change this running study): validate SSW,
GA and walker option objects before loading the calculator or doing initial
quenches; reuse the immutable option objects. CPU1464039 passes all7 runner
checks, including the original invalid MC field producing an archived zero-PES
failure without creating a calculator. No core search algorithm changes. Current
C60 jobs still execute the frozen ec01115 runner with corrected v2 plans.
Combined offline audit1463669 depends on all four corrected C60 jobs and requires
all8 planned Cu13/C60 result summaries; missing arms cannot silently pass.

Interim seed3 audit1464340: both corrected C60 arms close at60000 requests,
with1281 shared initialization requests. Fresh best energies are
-62189.39215324536eV (GA) and-62185.74784164229eV (independent SSW), a3.6443eV
GA advantage for this seed only. Neither meets the target cage or reference
energy criterion. Both force/composition/PBC checks pass; GA's nonperiodic
bookkeeping cell differs as anticipated. Second seed remains running; no
parameter, budget, model or input change follows this interim result.
