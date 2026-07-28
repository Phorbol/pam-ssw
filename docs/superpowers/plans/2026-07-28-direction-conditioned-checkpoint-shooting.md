# Direction-conditioned Checkpoint Shooting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether fixed C60 direction-conditioned walks fail because no productive checkpoint is reached or because the current uphiller propagates past a productive checkpoint.

**Architecture:** Reuse PAM-SSW's existing per-macro-step proposal-relaxation trajectory files. A run-local script generates 12 frozen uphill trajectories, extracts the final frame of every macro step, strictly quenches every checkpoint through an independently accounted walker, and builds descriptive evidence without changing `pamssw/`.

**Tech Stack:** Python 3.12, NumPy, ASE trajectory I/O, existing PAM-SSW/MACE calculator, pytest, JSON evidence.

---

### Task 1: Lock the pure experiment contract

**Files:**
- Create: `tests/unit/test_direction_conditioned_checkpoint_shooting.py`
- Create: `runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py`

- [ ] **Step 1: Write failing tests for the case matrix and classification**

Test the exact 12-case Cartesian product and the four classifications:

```python
def test_case_matrix_is_the_preregistered_twelve_cases():
    module = _load_module()
    matrix = module.case_matrix()
    assert len(matrix) == 12
    assert matrix[0] == {
        "state_id": "intermediate_accepted",
        "seed": 42,
        "arm": "balanced_refinement",
    }
    assert matrix[-1] == {
        "state_id": "plateau_accepted",
        "seed": 44,
        "arm": "deep_refinement",
    }


@pytest.mark.parametrize(
    ("productive", "expected"),
    [
        ([False, True], "productive_final"),
        ([True, False], "overshoot"),
        ([True, True], "productive_earlier_and_final"),
        ([False, False], "no_productive_checkpoint"),
    ],
)
def test_classification_uses_checkpoint_order(productive, expected):
    module = _load_module()
    checkpoints = [
        {"step_index": index, "productive": value}
        for index, value in enumerate(productive, start=1)
    ]
    assert module.classify_trajectory(checkpoints) == expected
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_direction_conditioned_checkpoint_shooting.py
```

Expected: import or attribute failure because `run_audit.py`, `case_matrix`, and
`classify_trajectory` do not yet exist.

- [ ] **Step 3: Implement only the pure contract**

Define:

```python
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS = {
    "balanced_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    },
    "deep_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
}
MEANINGFUL_ENERGY_DROP_EV = 0.001
```

`classify_trajectory` must reject an empty checkpoint sequence, require
one-based consecutive `step_index` values, and classify from the stored
`productive` booleans without recomputing them.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_direction_conditioned_checkpoint_shooting.py \
  runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py
git commit -m "test: lock checkpoint shooting contract"
```

### Task 2: Lock evidence and accounting semantics

**Files:**
- Modify: `tests/unit/test_direction_conditioned_checkpoint_shooting.py`
- Modify: `runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py`

- [ ] **Step 1: Write failing tests for evidence closure**

Construct 12 synthetic case rows with:

- a closed generation purpose ledger;
- direction FE equal to 24 times the direction-selection count;
- two ordered checkpoints;
- a closed independent quench ledger for every checkpoint;
- explicit certificate, new-basin, landing-energy, and productive fields.

Assert that `build_evidence` reports:

```python
assert evidence["cohort"]["completed_cases"] == 12
assert evidence["cohort"]["checkpoint_count"] == 24
assert evidence["classification_counts"]["overshoot"] == 12
assert evidence["production_default_changed"] is False
assert evidence["meaningful_energy_drop_threshold_eV"] == pytest.approx(0.001)
```

Then corrupt one generation ledger, one checkpoint quench ledger, one
direction-selection FE count, and one checkpoint sequence in separate
assertions; each must raise `ValueError`.

- [ ] **Step 2: Run the tests and confirm RED**

Expected: failure because `build_evidence` is absent.

- [ ] **Step 3: Implement minimal validation and aggregation**

`build_evidence(rows)` shall:

1. require the exact 12 unique case keys;
2. require at least one checkpoint per case;
3. validate consecutive checkpoint indices;
4. close generation and checkpoint purpose ledgers;
5. validate `direction_oracle == 24 * direction_selection_count`;
6. require boolean certificate and productive fields;
7. recompute the expected classification from checkpoint order;
8. aggregate classifications, certificate counts, checkpoint counts,
   meaningful checkpoint counts, and force-evaluation totals by state and arm;
9. retain every raw case and checkpoint in the evidence payload.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_direction_conditioned_checkpoint_shooting.py \
  runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py
git commit -m "test: close checkpoint shooting ledgers"
```

### Task 3: Implement the frozen CUDA runner

**Files:**
- Modify: `tests/unit/test_direction_conditioned_checkpoint_shooting.py`
- Modify: `runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py`

- [ ] **Step 1: Write failing tests for checkpoint discovery**

Create temporary files named:

```text
trial0001_proposal001_step001_proposal_relax.xyz
trial0001_proposal001_step002_proposal_relax.xyz
trial0001_proposal001_step003_proposal_relax.xyz
```

Assert that `discover_checkpoint_paths(directory)` returns them in step order.
Assert that a missing step, duplicate step, unexpected trial/proposal, or empty
directory raises `ValueError`.

- [ ] **Step 2: Run the tests and confirm RED**

Expected: failure because `discover_checkpoint_paths` is absent.

- [ ] **Step 3: Implement checkpoint discovery and execution**

The runner shall reuse:

- `_load_fixed_states` from
  `runs/20260728-block-krylov-fixed-starter-escape/run_ablation.py`;
- `validate_direction_trace` from
  `runs/20260728-block-krylov-direction-gpu-ablation/run_ablation.py`;
- `SurfaceWalker._proposal_pool` for one frozen uphill trajectory;
- existing `write_relaxation_trajectories=True` with stride 1;
- `SurfaceWalker._clip_walk_displacement` to transform every raw optimizer
  last-frame into the effective macro checkpoint;
- `SurfaceWalker.relax_true_minimum` configured with ASE-LBFGS primary,
  ASE-FIRE fallback, `fmax=0.01 eV/A`, and 400 iterations.

For each checkpoint:

```python
checkpoint_atoms = ase.io.read(path, index=-1)
checkpoint_state = State(
    numbers=np.asarray(checkpoint_atoms.numbers, dtype=int),
    positions=np.asarray(checkpoint_atoms.positions, dtype=float),
    cell=starter_state.cell,
    pbc=starter_state.pbc,
    fixed_mask=starter_state.fixed_mask,
    metadata={"checkpoint_path": str(path)},
)
checkpoint_state, clipped = SurfaceWalker._clip_walk_displacement(
    starter_state,
    checkpoint_state,
    config.walk_trust_radius,
)
```

Use a new accounted `SurfaceWalker` for each checkpoint. Evaluate its true-PES
energy under `ESCAPE_TRUE_PES_CHECK`, run the strict quench, add the landing to
a fresh two-state archive seeded only with the shared starter, and record the
independent purpose ledger.

After each completed case, atomically rewrite `raw.json`. After all 12 cases,
write `evidence.json`. Pin the full execution commit and reject a tracked dirty
worktree before model loading.

- [ ] **Step 4: Run unit tests and static verification**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_direction_conditioned_checkpoint_shooting.py
/root/miniforge3/envs/mace_les/bin/python -m py_compile \
  runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py
git diff --check
```

Expected: all tests pass; compilation and diff checks exit zero.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_direction_conditioned_checkpoint_shooting.py \
  runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py
git commit -m "exp: add checkpoint shooting runner"
```

### Task 4: Run and interpret the production mechanism audit

**Files:**
- Create: `runs/20260728-direction-conditioned-checkpoint-shooting/output/**`
- Create: `runs/20260728-direction-conditioned-checkpoint-shooting/conclusion.md`

- [ ] **Step 1: Run the pinned CUDA experiment**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/pamssw-matplotlib \
  /root/miniforge3/envs/mace_les/bin/python \
  runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py \
  --output-dir runs/20260728-direction-conditioned-checkpoint-shooting/output \
  --expected-git-commit "$(git rev-parse HEAD)"
```

Expected: 12 completed trajectory cases, all generated checkpoints preserved,
and a closed evidence file. If CUDA is hidden by the sandbox, rerun the exact
command with the already authorized GPU escalation.

- [ ] **Step 2: Verify evidence**

Run a read-only verifier that checks:

- exactly 12 unique cases;
- every trajectory has consecutive checkpoint indices;
- every source path hash closes;
- every purpose ledger closes;
- every completed direction selection costs exactly 24 force evaluations;
- classifications recompute from raw checkpoints;
- production defaults remain unchanged.

- [ ] **Step 3: Write the mechanism conclusion**

Report, without automatic promotion:

- trajectory classification counts by state and arm;
- earliest and final productive checkpoint frequencies;
- checkpoint landing-energy curves and strict-quench costs;
- whether productive intermediate checkpoints are later lost;
- whether plateau success is present before the final deep checkpoint;
- exact generation and diagnostic shooting costs;
- the claim ceiling and any CUDA replay sensitivity.

- [ ] **Step 4: Run focused regression verification**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_direction_conditioned_checkpoint_shooting.py \
  tests/unit/test_block_krylov_fixed_starter_escape.py \
  tests/unit/test_block_krylov.py
git diff --check
```

- [ ] **Step 5: Commit, push, and update PR #10**

```bash
git add runs/20260728-direction-conditioned-checkpoint-shooting
git commit -m "exp: record checkpoint shooting audit"
git push pam feature/posterior-terminal-outcome-validation
```

Update PR #10 with the evidence-backed direction-versus-uphiller conclusion.
