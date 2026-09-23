# C4H6 MH-1 fixed-protocol coverage comparison

Development comparison, frozen before these runs. This tests the existing
fixed-cell SSW and two LS implementations on one molecular reaction landscape;
it cannot establish cross-system generality or reproduce PBE chemistry.

## Question and competing explanations

Does local softening improve discovery of force-qualified structures with
changed bonding at comparable E/F cost? An alternative is that it mainly
changes geometry/softening response and spends calls returning to butadiene.
Native LS may also need substantially more updates than paper LS to activate;
equal target values do not imply equal instantaneous response.

The lifecycle pilot (seed59, job1469509) finished all 36 attempts and all39
fresh checks. Search requests were7651/7512/8108 for SSW/paper/native LS,
with111/108/115 seconds respectively. Linear extrapolation suggests roughly
250–270k requests and one GPU-hour per400 attempts, but changing chemistry may
change cost. Pilot outcomes are development evidence, excluded from this
comparison. No algorithm parameter is selected from its structural outcomes.

## Frozen protocol and cost limits

Three methods × seeds61,67; each starts independently from the same qualified
G2 trans-butadiene input and pays its own initial quench. Model, geometry,
optimizer, rotation, Gaussian, temperature and LS settings are unchanged from
the pilot; effective configuration is exported and checked against it.
The only extensions are new RNG seeds and a longer run.

Each arm:400 outer attempts OR320000 E/F requests OR6600 search seconds,
whichever comes first; at most401 independent fresh checks. One V100 per job,
hard wall2h, at most2 concurrent jobs; six arms total at most1920000 search
plus2406 fresh requests and12 GPU-hours. No automatic retries or budget extension.
Checkpoint completed outer attempts using the existing interface. Large raw
ledgers/checkpoints stay on shared storage; small protocols/reports stay inGit.

LS paper §3.1 visits400 minima, not necessarily400 attempts. We report actual
returned force-qualified structures separately, including repeats. A truncated
run or failed quench does not count as a visited minimum. We do not run until
400 unique minima, nor silently change the stopping rule to match the paper.

## Analysis fixed before results

- Reconcile initial + all outer-attempt costs against the request ledger.
  Include failed quench, LS preparation and rejected attempts. Report actual
  calculator invocations separately from E/F requests and cache reuse.
- For each seed/method, report coverage versus cumulative requests. The primary
  common-cost checkpoint is200000 requests, using only fully completed landings
  before that limit. If an arm does not reach it, label that comparison censored;
  do not select a new favorable prefix after reading results.
- Also report the400-attempt endpoint or explicit cap, with its actual costs;
  different endpoint costs are not an equal-cost efficiency ranking.
- Graph identity uses the existing element-labeled graph/isomorphism helper.
  Apply the same class mapping across arms. Report connected components and
  formulas; H2 products are retained because they are relevant to the paper.
  Bond graphs alone do not specify bond order, radicals or electronic state.
  Connected-structure classes and dissociated-product classes must be reported
  separately; total graph novelty cannot substitute for either. The pilot has
  examples of C4H5+H, C2H3+C2H3 and C2H4+C2H2, so this separation is necessary
  to avoid calling more fragmentation better isomer discovery.
- For butadiene-like graphs report raw CCCC torsion and existing cos-sign regions
  separately; cis/trans geometry is not new connectivity or a certified minimum.
- Independently reevaluate every returned structure with a fresh calculator:
  fmax≤0.03eV/Å, matching composition/cell/PBC and reported energy. Force
  qualification is not Hessian stability or chemical validation.
- Report LS response histories in eV/atom, native update cycle and all failures.
  The Gaussian-limit stop reason is distinct from a failed true quench.
- Report both seeds individually. Two seeds estimate neither a precise success
  probability nor general LS superiority. No pooled score with C60/materials.

## Decision rule

If the numerical chain fails, diagnose its first failure before further search.
If qualified connectivity discovery improves at the common cost in both seeds,
retain LS as supported on this model/task and test a separate existing task;
do not promote a universal default. Mixed or null results do not trigger a
C4H6 parameter sweep. Investigate the already recorded mechanisms and then
return to the broader fixed-cell evidence matrix. Any apparent reactive product
remains a candidate until electronic/physical checks appropriate to that claim.

Source: Guan, Shang & Liu, JCTC2024, DOI10.1021/acs.jctc.4c01081,
§3.1/Fig.5 and SI; the model differs from its PBE oracle.
