# Paper-LS checkpoint metadata repair

Observed defect: the archived MH-1 pilot has12 paper-LS response observations,
but its checkpoint response has `steps=0` and `last_response=None`. The generic
dataclass copy used `replace` with init fields only, resetting these init=False
fields. Native LS uses a different runtime class and was not affected.

Minimal repair: explicitly deep-copy LSResponseState in `_checkpoint_copy`.
No schema, public API, optimizer or adaptation-formula change. Frozen amplitudes
were already preserved; neither affected metadata field enters the next LS
strength calculation. Previously lost historical counts are not reconstructed.

CPU1470000 reproduced the defect on the existing Cu2/EMT continuous-versus-split
checkpoint test: expected2 updates, saved0. CPU1470014 then passed21 targeted
tests covering checkpoints, recovered-rotation LS and pool restarts. An archived
old MH-1 checkpoint still loads with next_index12 and its original missing
metadata. `sacct` confirms FAILED(1)/COMPLETED(0), CPU-MISC allocationdpn01,
15s/4s; the shell hostname printed in the log is not used as allocation evidence.

Core/test commit: c047cd914acfde8c787eff9d349aa500036be323. The integration branch
is intentionally unchanged while its frozen six-arm experiment executes.
This repairs state fidelity; it is not an algorithm-performance improvement.

Additional pool-persistence regression CPU1470285 passed both existing uniform
and PAM adapter cases (2 passed in 1.15s):

```sh
env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD" /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q \
  tests/standalone/test_pool_checkpoint_real.py::test_real_ls_pool_resume_matches_uninterrupted
```

Logs: `pool-1470285.out`, `pool-1470285.err`. This checks the existing schema5
pool route in addition to the earlier core checkpoint checks; no new PES
performance experiment was run.
