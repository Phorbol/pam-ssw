# Native-LS pool worker CPU preflight

This is an API, serialization, and accounting smoke test for the research worker. It is not an MH-1/omol qualification or an SSW effectiveness result. The worker remains an experimental runner pending the real-model review and authorized CPU/GPU run.

The final preflight reused the qualified `Icosahedron('Cu', 2)` Native-LS EMT fixture from `tests/standalone/test_pool_checkpoint_real.py::test_real_native_ls_mc_pool_resume_matches_uninterrupted` and its `_native_config` / `_native_ls` helpers in `test_ls_pool_restart.py`. The temporary plan used its two-step Native-LS/Native-MC settings and an ASE-permutation pool adapter. Production plan files were unchanged.

Job 1500664 ran on CPU-MISC with `rush-cpu`, account `sjtu-caoxiaoming`, and a five-minute limit. Slurm reports `COMPLETED`, elapsed 9 seconds, one task requested and two CPUs allocated. It first ran the analyzer regression command from the LS-mechanism checkout; the result was `3 passed in 0.64s` in [analyzer-pytest.txt](analyzer-pytest.txt). It then ran MC, uniform-pool, and PAM-pool arms sequentially through the same worker path.

| Mode | Status | Outer records | Returned landings | Search E/F | Fresh E/F | Fresh force-qualified | Selector decisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| MC | completed | 2 | 2 | 67 | 3 | 3/3 | — |
| Uniform pool | completed | 2 | 2 | 67 | 3 | 3/3 | 2 |
| PAM pool | completed | 2 | 2 | 64 | 3 | 3/3 | 2 |

The fresh denominator is the initial structure plus both returned landings in each arm. The verifier loaded each `checkpoint.pkl` with the existing `load_ssw_checkpoint` API and confirmed Native-LS runtime and frozen bias state were present. Search and fresh paid counts matched their ledgers; each search result satisfied `initial.evaluation_requests + sum(record.evaluation_requests) == result.evaluation_requests`. Pool adapters recorded two decisions apiece. MC had no runtime pool selector; its archive identity readout was offline.

The failed preflight attempts are preserved and counted:

- Job 1500653 failed before calculator construction because the worker rewrote the temporary input path before reading it. No E/F request was made.
- Job 1500659 used an invented Cu4 fixture whose Cu bond cutoff contained no bonds. All three arms stopped at Native-LS initialization after 5 search requests plus the independent initial fresh check per arm: 18 paid E/F total, no outer landing, and no selector decision.
- Job 1500660 used the validated Cu13 fixture and completed the MC arm's two outer steps and three fresh checks (67 search + 3 fresh E/F), but JSON serialization failed on `NativeLSRuntime` inside the checkpoint. That result led to the current minimal design: persist the checkpoint with the existing checkpoint API and serialize only SSWResult fields excluding `checkpoint`. Pool arms were not reached.

All final artifacts are in [outputs-checkpoint-1500660](outputs-checkpoint-1500660/), including per-arm effective plans, actual configurations, serialized SSW results with checkpoints saved separately, ledgers, fresh certificates, structures, and pool reports. The submitted commands are retained in [run-checkpoint-1500660.sbatch](run-checkpoint-1500660.sbatch); the verifier is [verify.py](verify.py). Earlier failed scripts and outputs remain alongside these final artifacts.

The CPU preflight establishes that the worker exercises the Native-LS, independent-fresh, and pool-selection code paths and closes its request accounting on this EMT fixture. It does not establish behavior on the planned molecular inputs or MH-1/omol, and it makes no search-quality claim. No GPU job was submitted.
