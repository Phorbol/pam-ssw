# Approved optional molecular pool identity — 2026-09-26

User approved `ase_permute_v1` while preserving the historical default. This
implements the existing proposal, not a new scientific search strategy.

Changes: research-only archive subclass replaces its distance calculation with
`ase.geometry.distance(..., permute=True)/sqrt(N)`. It inherits insertion, visits,
prototype and duplicate bookkeeping. Composition must agree; only nonperiodic,
unconstrained states are supported. A one-atom isolated structure has no internal
geometry after removing translation. Input and representative atom orders stay
unchanged. Core SSW, RDF and PAM scoring are untouched.

`PoolStarterAdapter(..., identity_matcher='ordered_v1')` is still the default and
keeps exactly the historical v1 checkpoint contract shape. Opt-in
`identity_matcher='ase_permute_v1'` uses v2 and records the ASE version, callable,
normalization and geometry domain. Restore uses the same archive subclass and
rejects mode/version disagreement before mutation. No old archive is rematched.

Verification:
- CPU1500564: nine new tests fail because the new constructor argument is absent
  (expected red, before implementation).
- CPU1500567: 34 tests pass in2.10s. Command: `python -m pytest
  tests/standalone/test_pool_ase_identity.py
  tests/standalone/test_pool_starter_adapter.py
  tests/standalone/test_pool_checkpoint.py
  tests/standalone/test_pool_checkpoint_real.py -q` in mace_env, CPU-MISC.
- These include end-to-end EMT LS uniform/PAM continuous versus resumed runs,
  input-order preservation, duplicate credit, legacy/default round trip,
  cross-mode/ASE-version rejection, and periodic-input rejection. Existing EMT
  fixtures keep their historical tolerances; they are interface checks, not a
  scientific efficiency experiment.

Independent whole-change static review found no blocking issue. The inherited
adapter performs a find_match before add, which repeats the geometry comparison;
this existing cost remains, without double bookkeeping. Measure before considering
an optimization; no core archive refactor is part of this change.

CPU1500597 completed the actual research-adapter replay of the saved C4H6/C60
geometry panel (17s, zero PES; LS evidence branch ce61954). Root independently
checked all330 opt-in rows: entry_count1, mapping[0,0], duplicate success credit0,
representative order preserved, and continuous/restored states equal. Legacy
v1 restoration and incompatible-mode/version rejection also passed.
[Reproducible panel](https://github.com/Phorbol/pam-ssw/tree/ce61954/research/ga_ssw/evidence/pool-ase-adapter-20260926).
No GPU or new MLIP search was submitted.
Prior ASE geometry qualification was330/330, but approximate matching remains a
geometric tolerance decision, not proof of common physical basins or globally
minimal permutation RMSD. No algorithm-performance/default-promotion claim.

Source `git diff --check -- *.py *.md` passed. The original red pytest log
contains pytest-generated trailing whitespace; it is retained verbatim as raw
evidence, not represented as a successful whole-artifact whitespace check.
