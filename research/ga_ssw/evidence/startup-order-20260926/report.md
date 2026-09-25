# Optional randomized startup: implementation qualification complete

User approved the bounded design on2026-09-26. Implementation bf70b21 changes only `pamssw/standalone/recovered_direction.py`: append `startup_order='legacy'`, allow opt-in `'randomized'`, and use a single shared permutation during initialization. Reference coordinates and current ASE atoms (including species) are permuted together, then active and diagnostic pair/group indices are mapped back. Ordinary landing refresh remains unchanged. Pool jumps already call initialization, while checkpoint resume restores state without drawing another permutation. No new numeric threshold, optimizer, Gaussian rule or checkpoint schema.

Usage with existing settings:

```python
from dataclasses import replace
settings = replace(settings, startup_order='randomized')
# run_ssw(..., recovered_direction=settings)
```

Legacy mode preserves iterator RNG compatibility; randomized mode needs `permutation()` (the normal run_ssw NumPy Generator has it). A missing field in old pickles resolves to legacy via the dataclass default. Changing the explicit policy while resuming is rejected before RNG restoration or PES calls. New-mode files require code that understands the new option; no claim of running them in older software.

## Verification

- New tests first failed on the absent field/API (three intended failures). Two subsequent checkpoint test failures were fixture errors: no callback had requested a returned checkpoint. Adding the existing progress callback fixed the fixture without modifying the algorithm or acceptance criterion.
- Final command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q tests/standalone/test_recovered_startup_order.py tests/standalone/test_recovered_direction.py tests/standalone/test_recovered_direction_checkpoint.py tests/standalone/test_pool_direction_checkpoint.py`. Result:24 passed; [log](contract-tests.log). Includes C60 and mixed-species butadiene mapping, synthetic old schema4/5 files, policy mismatch, unchanged landing refresh and pool restart mapping.
- CPU1493366, authorized group account/CPU-MISC/rush-cpu, completed exit0 in1m31 under the10-minute/40000-request bound. Frozen source bf70b21; no GPU or MACE search. Actual **2328 search requests +6 independent EMT checks**. [Protocol](plan.md), [runner](qualify.py), [summary](runs/summary.json).
- Actual pre-feature archived C60 schema4 checkpoint (not a synthetic new file) loads with its missing field as legacy and restores zero outer steps with **zero PES requests**, retaining RNG. [Archive check](runs/old-schema4.json). Synthetic schema5 old-field behavior is unit evidence, not a claim to have replayed a historical full-direction schema5 production trajectory.
- Real Cu13/EMT: uninterrupted two steps, initialization pause/resume, and outer-step pause/resume match exactly in each of two groups: ordinary walking (404 requests each) and explicit pool jumps (372 each). Records, minima, current/best geometry, direction state, main/selector RNG, pool state and costs agree. Actual pool-restart branches execute. Every independently evaluated best endpoint satisfies0.03eV/A; max force0.00723 or0.01164eV/A. Large result snapshots remain under `runs/` rather than inGit.
- Independent bounded read-only code review found no defect in mapping, legacy random draws, missing-field behavior or restart scope. It did not rerun experiments; this is engineering review, not additional scientific evidence. Parent checked actual outputs and scheduler status.

## Meaning and stopping decision

This closes the approved interface change. It removes systematic input-label preference from the startup selector **in distribution under uniform permutations**. It does not enforce identical same-seed trajectories after relabeling, prove full-walker equivariance, or improve the C60 success rate. The earlier2/4 versus0/4 C60 probes do not isolate startup tie-breaking: the physical random directions also change. No extra C60, permutations, parameter search or production budget follows from these tests. Keep legacy default and the new mode explicitly opt-in; close this subtask and retain the broader SSW/LS effectiveness gaps.
