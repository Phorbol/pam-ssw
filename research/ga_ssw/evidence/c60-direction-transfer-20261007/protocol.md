# Random C60: transfer of the existing direction bundle

Decision question: do the already implemented local pair/group selection and
continuous displacement memory improve target discovery at a fixed total cost
on genuinely new random carbon clouds? LJ55 showed a scoped positive signal,
M80 showed mixed progress/no target, and local C60 defect repair cannot establish
random-start transfer. Historical seeds17093–17096 and17101/17102 have been used;
none is reopened. This is a development comparison, not a success-rate campaign.

Only intentional factor: recovered_rotation versus recovered_direction. Their
pre/max rotations, residual stopping, Euclidean metric and40-request cap are
identical. The full bundle uses the existing full_peratom recipe from the
LJ55/M80 comparison: ratio_local50, local_probability/group_threshold .5,
c1_radius_policy per_atom, legacy startup, nonperiodic geometry. This compares
a bundle, not a claim about one selector or memory term. No LS, pool, Q modes,
adaptive Gaussian, confinement, temperature change or optimizer replacement.

Two new generator/search seeds26100781/26100782, checked against the current
research index. Same raw coordinates per pair. Reuse the documented finite
sphere initialization: uniformly draw from[-5,5]^3, reject outside5A sphere and
within1A of any already accepted atom;60C, center translated25A,50A diagonal
cell, nonperiodic. These are provisional input-distribution choices retained
from prior random C60 protocols, not literature-optimal parameters. Record
trials and actual inputs; no rejection/replacement based on relaxed quality.

Freeze cached MH-1 model a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47,
omol/float64/CUDA with cueq/oeq disabled. Reference is the independently
qualified Ih at c60-local-defect-20260925/qualification/isomer-1/final.extxyz,
fresh E=-62215.393370790625eV, fmax .00407518eV/A (source result.json).
Reuse existing local curvature qualification, not a new Hessian campaign:
c60-local-defect-20260925/curvature/results.json records174 internal eigenvalues
for this reference, all positive (minimum3.03193466eV/A²), with the two lowest
directions independently checked at .02A. Its runner uses the same MH-1/omol
float64 backend. This supports a locally stable cage target on that model,
not global-minimum/DFT validity; the old365 reference E/F calls remain historical
costs, not newly issued requests or independent transfer evidence.
SSW settings inherit the existing C60 protocol: width .6A,NG12,T150K,
fmax .03eV/A,bias_fmax .1,1000 inner iterations,fd .001A,
Safe-total/history500,direction_only,NativeMC(.1eV,99999). Preserve these
as experimental choices, not promote them to defaults or fit on new seeds.
NativeMC retains its recovered divisor20;150K here is a search configuration,
not a claim of canonical150K sampling (effective exponent scale is20kBT).

Stages: first generate/freeze inputs and snapshot the selected core/runner,
then verify the existing harness on CPU (including budget and checkpoint
regressions). Qualify each random input through the ordinary initial true
quench under the same model/settings, independently cold-check it and the
reference. Initial failures are retained with costs and never replaced.
Only after this qualification and source review may the fixed four-arm search
be submitted. Search consumes the raw cloud, so initial quench remains charged
to each arm; qualification costs are separate experimental overhead.

Qualification cap:3000E/F per input,100seconds perinput; one reference cold
plus one cold per certified input,6000search+3cold maximum,oneV100/5min,
240second process deadline. No expansion on failure. Actual preparation
requests and runtime are retained even if the subsequent comparison cannot run.

Search cap:60000 paid requests per arm including initial quench;240000total,
at most3fresh perarm (initial,best,one separate cage candidate), at most2V100
concurrently. One80minute allocation perarm; total<=5h20GPU allocation time.
Hard request/time limits, no automatic resume, retry, new seed or extension.
Expected ordinary cost comes from prior60k MH-1 runs (~52–54minutes/arm),
not model single-point timing or paper budgets.60000 does not guarantee a cage;
it supports a matched early-search transfer/cost test. Four comparisons or
two successful seeds do not estimate a robust success probability.

Report all qualified candidate minima, failures and cumulative costs at common
paid prefixes15k/30k/60k; number of outer steps is secondary. Structural target:
connected60C,3regular/3connected planar graph,12five- and20six-member faces,
checked at1.64/1.70/1.80A and with existing3D extent sanity check. Report cage,
energy<=reference+.01eV, and their conjunction separately; independently cold
verify selected candidates at fmax .03. A lower-energy fragmented structure is
not a success. Minimum identity/near-target counts remain diagnostics, not a
surrogate for this target. Physical force alone does not prove Hessian stability.

Decision: paired target hits or consistent better cost-to-qualified target
justify retaining the bundle as a candidate for a later independent broader
test. Mixed/no-hit outcomes remain inconclusive for global efficiency and do
not trigger scale-up or tuning. The benefit boundary is this MH-1 PES and
input distribution, not the paper's carbon potential or native LASP ranking.
SSW/LS/GA papers supply the C60 target; current binary evidence supplies the
direction mechanism, with explicitly incomplete native-state parity. Sources:
docs/research/2026-10-07-lasp-mainline-assessment.md,
docs/research/2026-09-24-c60-long-budget-decision.md,
research/ga_ssw/evidence/lj55-direction-target-20261007/decision.md.
