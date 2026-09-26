# TYPE4 runner correction before physical search

GPU1499057 (code04cff29) reached zero E/F calls in both arms: the custom Oracle
omitted the counted-surface `requests` property required by run_constrained_ssw.
Original output, config and failure summary remain in type4-runs/ and type4-1499057.out.
The sole runner correction exposes live+replayed request count. No core change.

CPU1499094 reproduced the missing-property error. CPU1499105, and its persistent
recording repeat1499116, passed the count interface but encountered the existing
native acos finite-domain exit on an isotropic harmonic fixture. The identical
Hessian makes Hv collinear; this is a known native arithmetic boundary, not a
MACE or periodic geometry result. Raw persistent synthetic artifacts are in
runner-check-v2/. We did not clip acos or change production criteria.

A nondegenerate positive diagonal harmonic fixture (stiffness1..3, test only)
then exercised the full script, independent endpoint audit and split1+1 replay:
CPU1499122 passes1test; both arms qualify and six endpoint checks pass. This is
runner wiring qualification only. Artifacts are in runner-check-v3/.

The actual MACE test is resubmitted to a new type4-runs-v2/ directory after this
precheck, with unchanged physical input/config/seed/8000-search-request limit,
oneV10020min. Prior failed launch consumed zero search calls. No solver/budget
failure is retried or tuned. A script-level error before search justifies this
corrected launch; it is not an algorithm performance observation.
