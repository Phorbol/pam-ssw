# Fixed-cell SSW completed-attempt pause

Implemented the first, already approved part of the GA active-walk proposal:
optional `checkpoint_callback` in fixed-cell SSW, paper LS and native LS.
GA v2/active-walk remains deferred. This adds safe outer-attempt pauses for
long fixed-cell runs, with no new search formula, scientific parameter or
checkpoint schema. A true callback return yields result.status='paused'; its
completed checkpoint remains resumable. Terminal failures do not invoke the
callback. A supplied checkpoint_path is saved before callback invocation.
Callback receives a detached copy; its mutations cannot alter the live walker.
Without callback/path there is no added per-step snapshot copy.

Implementation9a3c27d; budget-regression7893126; merged3a8313a. Root reviewed
actual code and logs. Integration and tested branch have identical pamssw tree
c91783c10ae061f86cfadee08e3c5ed1dcfa7263 and tests tree
dfa169b977f62890cc60d028e395c8591cf62cd0, so no unchanged rerun was added.

CPU1473311 first reproduced the absent API (expected TypeError). CPU1473333
then had one test-fixture error: a continuous run without checkpointing returns
checkpoint=None. Enabling a non-pausing callback in that comparison fixed the
test; production code was unchanged. CPU1473337 passed32 targeted tests:

```sh
env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q \
  tests/standalone/test_ssw_boundary_pause.py \
  tests/standalone/test_ssw_checkpoint.py \
  tests/standalone/test_pool_checkpoint_real.py \
  tests/standalone/test_ls_pool_restart.py \
  tests/standalone/test_recovered_rotation_ls.py
```

Workdir was `/home/gengjianrui/bin/pam-ssw-worktrees/ssw-boundary-pause`.
CPU1473367 reran only `tests/standalone/test_ssw_boundary_pause.py` after adding
the budget-refusal regression:8passed. The retained1473363 log also has8passes
for that same file; this repeat is not independent evidence. This is not40
distinct tests. All jobs
ran on dpn01/CPU-MISC; failed logs remain alongside successful logs. Existing
Cu/EMT tests check actual continuous/split structures, landing order, RNG,
request cost, LS state, pool state, snapshot isolation and failure semantics.
These are implementation checks, not a search-performance claim.

The extra regression confirms that budget refusal after a safe boundary leaves
a terminal, non-resumable diagnostic checkpoint and cannot be disguised as a
pause. Loading an earlier completed boundary is structurally possible; it never
grants extra budget or refunds already paid work. External run accounting must
retain any cost after that boundary. The callback cannot interrupt a Gaussian
or local line search, and user callback side effects remain the caller's duty.

Not yet qualified: multi-hour MH-1 segmented execution and its cumulative
external ledger. The C60 large-budget proposal remains unsubmitted pending a
resource decision and this bounded preflight; do not equate these EMT tests with
C60 scientific acceptance.
