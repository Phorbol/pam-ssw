# Native kernel follow-up: rotation state and Gaussian call contract

This increment continues the binary + primary-paper + PAM comparison, without
claiming a complete native walker or introducing new search heuristics.

- `native_rotation_control.py`: unconstrained 40-degree cap, first-rotation
  FACT1 retry, strict stopping predicates, CBD_PreRot exception, evaluated n00
  rollback. Nine tests and an independent assembly review cover these rules.
- `native-gaussian-caller.md`: the height gate is exactly `climb_new`, transitioning
  to `climb_opt`. e2 is old bias energy only, tf0 is scratch rebuilt for old terms,
  and physical E is added later. Saved width is a measured displacement projection.
  Two consecutive old-bias force subtract passes need whole-function instruction
  validation. Do not wire a generic background E/F wrapper by assumption.
- `native_broyden.py`: only the recovered initialization/secant prefix, not a full
  optimizer. The dedicated prefix oracle executes original instructions through
  initialization and the second-call prefix; history-matrix update is not covered.
  The 50-column history matrix update is still being recovered separately.

The inproduct helper computes sum_i(sum_xyz(a_i)*sum_xyz(b_i)); it is not an
ordinary dot product. At the helper level the original instructions give 0 for
(1,-1,0) squared and 4 for a rotated vector (1,0,1), though both Euclidean squared
norms are 2. The prefix's zero normalizer gives nonfinite native DF/U. This is a
specific executable finding and a mathematical symmetry issue, not a measured
whole-search failure rate. It must not become an unlabelled PAM default.

Primary-paper linkage remains the archived BP-CBD 2012 and SSW 2013 texts;
`native-rotation-followup.md` distinguishes their direction-preservation intent
from executable constants and state transitions. Fresh author-index search found
CBD 2010 item50; author PDF retrieval timed out/failed TLS and ACS returned403,
so no new CBD2010 full-text equation is asserted. Author index:
https://zpliu.fudan.edu.cn/publication/list.htm.

Next dependencies: finish BRZERO4 matrix updates and restart semantics; execute
whole addgaussian with ng=1/2/3 and compare force to energy derivatives; recover
moveds retries and climb completion selection. Then build the explicit native
walker and replay a full escape with a common ASE energy/force backend. Existing
paper/Ritz and PAM implementations remain distinct baselines. No real-system
E2E or performance gain was established by this increment.

Verification after integration: `PAMSSW_NATIVE_MC_ELF=<uploaded lasp> python -m
pytest tests/standalone tests/reproduction -q` completed with **119 passed** in
4.25 seconds. Existing ASE/NumPy shape deprecation warnings remain. BRZERO4
prefix comparisons: 18/18, maximum DF error 2.22e-16 and U error 4.44e-16;
initial step and raw displacement matched exactly. No full Broyden matrix update
or full native escape has been validated. Work remains on the research branch;
no stable baseline changes, push or merge occurred.
