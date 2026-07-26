# pamssw

`pamssw` is a compact implementation of SSW and LS-SSW for fixed-cell atomistic
global optimization. It is designed for practical basin discovery: start from one
structure, walk on a biased potential surface, quench on the true potential, and
keep a deduplicated archive of minima.

The recommended user-facing route is `LS-SSW` with automatic local softening and
ASE-style relaxation. Use plain `SSW` mainly as a baseline or ablation.

## What It Does

- Runs SSW or LS-SSW searches from a single starting structure.
- Supports analytic toy potentials and any ASE-compatible calculator.
- Supports fixed-cell periodic systems through `cell` and `pbc`.
- Honors `fixed_mask` during ASE-based relaxation, useful for slabs.
- Builds an archive of accepted minima with energy and descriptor diagnostics.
- Can write accepted minima, all proposal minima, and relaxation trajectories as
  `.xyz` files.

## Current Scope

Included:

- Fixed-cell cluster, molecule, bulk, and slab searches
- ASE calculator bridge, including MACE through ASE-compatible factories
- MIC-aware fingerprints and MIC-aware LS-SSW local-softening pairs
- Automatic neighbor-pair local softening for LS-SSW
- Trust-region, archive-acquisition, and output-control diagnostics

Not included:

- Variable-cell search
- Transition-state refinement
- IRC or fake-IRC
- Canonical sampling or Metropolis-Hastings correction
- A full workflow wrapper for reading arbitrary structure files from the CLI

For periodic slabs, treat the current implementation as fixed-cell SSW/LS-SSW.
It can evaluate periodic energies and forces, but the workflow does not optimize
the cell.

## Experimental posterior-driven exploration core

`ExplorationController` is a generic, walker-independent synchronous batch
controller with the unchanged `uniform`, `posterior_proportional`, and
`minimal_ucb` policies. Here, **strategy-unbiased** has only an operational
strategy-support meaning: `uniform` and `posterior_proportional` give every
eligible starter strictly positive probability and log each action's exact
selection propensity. `minimal_ucb` is deterministic and does not provide
propensity-based full support. None of these terms claim thermodynamic,
kinetic, canonical, detailed-balance, or statistical unbiasedness. The current
legacy composite UCB remains the external/default SSW comparator; it is not
silently reimplemented or replaced.

The fixed Beta(1,1) posterior models the probability that an action lands in a
basin absent from the dispatch archive snapshot. Every planned action receives a
terminal attempt record only after its batch passes result-contract validation
and is finalized; malformed or non-`AttemptResult` returns, and returns for the
wrong action, fail closed before logging. Finalized failed attempts count as
false. Batches sample with replacement; repeated landings in the same novel
basin receive success credit for every action but produce one archive insertion.
The opt-in public `run_posterior_ssw` and `run_posterior_ls_ssw` functions
compose this core into fixed-budget SSW and LS-SSW campaigns. They accept a raw
`State`, bootstrap it with one true-PES quench, and charge both that bootstrap
and its final validation to the campaign's single total force budget. Every
dispatched action uses the same fixed action-force cap; all calculator calls
carry an evaluation purpose, and zero unattributed evaluations is a benchmark
eligibility condition.

The runner commits complete batches synchronously in slot order through one
`ThreadPoolExecutor`, then writes the compact action facts to
`events.jsonl`. The experimental `SSWAttemptWorker` remains a narrow
per-action boundary: every action creates a fresh calculator and
`SurfaceWalker`, derives an action-local configuration with `max_trials=1` and
the fixed action cap, and excludes internal proposal competition
(`proposal_pool_size=1` with duplicate rescue disabled).

`benchmarks/posterior_policy_compare.py` is a paired raw-fact
integration/ablation harness for the three policies. It records paired runs and
their accounting facts; it does not pick a winner, test statistical
significance, or claim performance superiority. Its tiny analytic `DoubleWell`
smoke is an integration check, not a scientifically valid policy benchmark.

Runtime validation is limited to the analytic backend and
`ThreadPoolExecutor`. This opt-in path has no recovery, resume, or replay;
no asynchronous racing; no MACE, GPU, or process-backend validation; and no
canonical, thermodynamic, or kinetic unbiased-sampling claim. It also makes no
policy-superiority claim. These are experimental controls for clean ablation,
not evidence of improved search performance.

## Install

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e '.[dev]'
pytest -q tests/unit tests/integration
```

## Recommended Production Presets

After extensive testing on C60 clusters and PdO slabs with MACE OMAT small, the
following parameters are recommended starting presets for `LS-SSW`.  The walker
envelope is system-dependent: cluster parameters should not be copied directly
to metal/adsorbate slabs.

### Cluster Walker

```
target_uphill_energy=0.8       # bias hill height (eV)
target_negative_curvature=0.05  # target curvature after bias flip
max_steps_per_walk=8            # bias accumulation steps per walk
max_step_scale=1.2              # trust-region upper bound on sigma
min_step_scale=0.1              # trust-region lower bound on sigma
proposal_trust_radius=1.5       # coordinate box during biased relax (Å)
walk_trust_radius=5.0           # per-atom displacement cap from seed (Å)
```

### Slab Walker

Start more conservatively for metal/adsorbate slabs where surface registration
must be preserved.  Loosen these only after checking displacement diagnostics
and accepted-minimum geometries.

```
target_uphill_energy=0.3-0.5    # lower hill height for slab registry
target_negative_curvature=0.05
max_steps_per_walk=4            # reduce multi-step momentum accumulation
max_step_scale=0.5              # avoid repeated sigma clipping at 1.2 Å
min_step_scale=0.05
proposal_trust_radius=0.75      # tighter biased-relax coordinate box (Å)
walk_trust_radius=1.5           # prevent cross-layer slab atom drift (Å)
```

### Local Softening (shared kernel, system-specific active count)

```
local_softening_penalty=buckingham_repulsive
local_softening_strength=0.15
local_softening_xi=0.3
local_softening_cutoff=2.0
local_softening_mode=active_neighbors

# C60 (60-atom cluster):
local_softening_active_count=3

# PdO slab (~115 atoms):
local_softening_active_count=5
```

### Optimizers

```
quench_optimizer=scipy-lbfgsb   # true-PES final relaxation
proposal_optimizer=ase-fire     # biased-PES proposal relaxation
proposal_relax_steps=80         # C60; PdO uses 300 for extra robustness
proposal_fmax=0.05
quench_fmax=0.01
```

### Archive and Selection

```
seed_selection_mode="archive_ucb"  # UCB bandit over all accepted minima
dedup_rmsd_tol=0.15             # structure deduplication (Å)
dedup_energy_tol=0.001          # energy deduplication (eV)
fragment_guard_factor=2.5       # reject walks that fragment the cluster
max_energy_drop_per_atom=5.0    # reject unphysical per-atom energy drops (eV)
```

Minimal Python example with ASE:

```python
import numpy as np
from ase.calculators.lj import LennardJones

from pamssw import LSSSWConfig, State, run_ls_ssw
from pamssw.calculators import ASECalculator

state = State(
    numbers=np.array([18, 18, 18]),
    positions=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
        ]
    ),
)

config = LSSSWConfig(
    max_trials=50,
    max_steps_per_walk=6,
    rng_seed=42,
    proposal_optimizer="ase-fire",
    quench_optimizer="ase-fire",
    local_softening_mode="neighbor_auto",
    accepted_structures_log="runs/example/accepted_structures.jsonl",
    accepted_structures_dir="runs/example/accepted_minima",
    write_proposal_minima=True,
    proposal_minima_dir="runs/example/proposal_minima",
)

result = run_ls_ssw(state, ASECalculator(LennardJones()), config)
print(result.best_energy, len(result.archive.entries), result.stats)
```

## Run From A Structure File

Use `read_state()` to read structure files supported by ASE, including `POSCAR`,
`.cif`, `.xyz`, `.extxyz`, and `.traj`. This is the recommended path for real
systems because calculator construction, MACE options, fixed atoms, and output
paths are usually project-specific.

Example runner:

```python
from pathlib import Path

from ase.calculators.lj import LennardJones

from pamssw import LSSSWConfig, read_state, run_ls_ssw, write_state
from pamssw.calculators import ASECalculator

outdir = Path("runs/my_structure_lsssw")
state = read_state("POSCAR")
config = LSSSWConfig(
    max_trials=50,
    max_steps_per_walk=6,
    rng_seed=0,
    proposal_optimizer="ase-fire",
    quench_optimizer="ase-fire",
    local_softening_mode="neighbor_auto",
    accepted_structures_log=str(outdir / "accepted_structures.jsonl"),
    accepted_structures_dir=str(outdir / "accepted_minima"),
    write_proposal_minima=True,
    proposal_minima_dir=str(outdir / "proposal_minima"),
)

calc = ASECalculator(LennardJones())
result = run_ls_ssw(state, calc, config)

outdir.mkdir(parents=True, exist_ok=True)
write_state(outdir / "best_minimum.xyz", result.best_state)
print("best_energy", result.best_energy)
print("n_minima", len(result.archive.entries))
```

`read_state()` preserves `numbers`, `positions`, `cell`, `pbc`, and ASE
`FixAtoms` constraints as `fixed_mask`. If you already have an ASE `Atoms`
object, use `state_from_atoms(atoms)`. To write a `State` back to any ASE
supported format, use `write_state(path, state)`.

For MACE, replace the calculator block:

```python
from mace.calculators import MACECalculator

calc = ASECalculator(
    MACECalculator(
        model_paths=["/path/to/mace-omat-0-small.model"],
        device="cuda",
        default_dtype="float32",
    )
)
```

## CLI Quick Start

The CLI is intentionally small and expects the structure to be written directly
in YAML. Use Python for file-based workflows, custom MACE setup, or richer output
post-processing.

```bash
python -m pamssw run-ls-ssw config.yaml
```

Example `config.yaml`:

```yaml
state:
  numbers: [18, 18, 18]
  positions:
    - [0.0, 0.0, 0.0]
    - [1.2, 0.0, 0.0]
    - [0.0, 1.2, 0.0]
  pbc: [false, false, false]

calculator:
  kind: ase
  factory: ase.calculators.lj.LennardJones

search:
  max_trials: 50
  max_steps_per_walk: 6
  rng_seed: 42
  proposal_optimizer: ase-fire
  quench_optimizer: ase-fire
  local_softening_mode: neighbor_auto
  accepted_structures_log: runs/example/accepted_structures.jsonl
  accepted_structures_dir: runs/example/accepted_minima
  write_proposal_minima: true
  proposal_minima_dir: runs/example/proposal_minima

output: runs/example/summary.json
```

The summary JSON contains:

- `best_energy`
- `n_minima`
- `archive_energies`

Accepted minima and proposal minima are written separately when their output
directories are configured.

## Slab And Periodic Systems

Periodic systems use the same `State` fields:

```python
state = State(
    numbers=numbers,
    positions=positions,
    cell=cell_3x3,
    pbc=(True, True, False),
    fixed_mask=fixed_bottom_layer_mask,
)
```

Use ASE-based optimizers for slabs:

```python
config = LSSSWConfig(
    max_trials=40,
    max_steps_per_walk=4,
    target_uphill_energy=0.5,
    max_step_scale=0.5,
    proposal_optimizer="ase-fire",
    quench_optimizer="scipy-lbfgsb",
    local_softening_mode="active_neighbors",
    local_softening_active_count=5,
    local_softening_penalty="buckingham_repulsive",
    local_softening_strength=0.15,
    local_softening_xi=0.3,
    local_softening_cutoff=2.0,
    proposal_relax_steps=300,       # PdO benefits from longer biased relax
    proposal_trust_radius=0.75,
    walk_trust_radius=1.5,
    accepted_structures_dir="runs/slab/accepted_minima",
    accepted_structures_log="runs/slab/accepted_structures.jsonl",
)
```

`ASECalculator` preserves `cell` and `pbc` when it builds ASE `Atoms`. Local
softening and pair-distance fingerprints use minimum-image distances on periodic
axes. The search remains fixed-cell: do not expect lattice-vector optimization.

**Slab-specific notes**: The shared softening kernel (`xi=0.3, strength=0.15`)
works for both C60 and PdO, but the walker envelope needs system-specific
tuning.  PdO used `active_count=5` (more softened atoms in the larger cell)
and `proposal_relax_steps=300` (longer biased relaxation in the denser atom
graph).  Metal slabs with weak adsorbate/surface registry should start with
the conservative envelope above; an Ir(111)-C18 smoke with the old cluster-like
envelope produced multi-Å top-layer buckling, so inspect
`step_displacement_*`, `proposal_relax_displacement_max`,
`true_quench_displacement_max`, and `walk_displacement_clips` before promoting
slab parameters.

## MACE Usage

Use MACE through the ASE calculator bridge. Construct the MACE calculator in
Python, then wrap it:

```python
from mace.calculators import MACECalculator
from pamssw.calculators import ASECalculator

mace_calc = MACECalculator(
    model_paths=["/path/to/mace-omat-0-small.model"],
    device="cuda",
    default_dtype="float32",
)

calc = ASECalculator(mace_calc)
result = run_ls_ssw(state, calc, config)
```

Keep MACE model path, device, dtype, and cuEq settings explicit in your runner.
The core `pamssw` package only sees an ASE-compatible calculator.

## Choosing SSW vs LS-SSW

Use `LS-SSW` for production searches. It adds local pair softening during the
proposal walk, which helps push structures out of the current basin while
rebuilding pair information from each seed structure.

Use plain `SSW` when you want:

- A baseline without local softening
- A faster smoke test
- An ablation against LS-SSW behavior

Python entry points:

```python
from pamssw import SSWConfig, LSSSWConfig, run_ssw, run_ls_ssw
```

CLI entry points:

```bash
python -m pamssw run-ssw config.yaml
python -m pamssw run-ls-ssw config.yaml
```

## Key Output Files

When configured, the search writes:

- `accepted_structures.jsonl`: one record per accepted new basin, including
  trial id, seed id, discovered entry id, energy, best energy, and descriptor.
- `accepted_minima/*.xyz`: accepted archive minima.
- `proposal_minima/*.xyz`: every proposal true-minimum, including duplicates and
  rejected proposals.
- `relaxation_trajectories/*.xyz`: proposal and true-quench trajectories when
  trajectory writing is enabled.

The returned `SearchResult` also exposes:

- `best_state`
- `best_energy`
- `archive.entries`
- `walk_history`
- `stats`

## Parameters Worth Tuning First

The presets above are starting points, not universal constants.  When adapting
to a new system, tune in this order:

1. `max_trials` — 40 for a smoke, 100 for a baseline, 500 for a deep search.
   C60 needed 100+ to reliably cross funnel boundaries.
2. `rng_seed` — run 3-5 seeds.  Seed variance on C60 is ~6 eV; single-seed
   conclusions are unreliable.
3. `local_softening_active_count` — how many atoms get softened per step.
   Scale with system size (C60 uses 3, PdO uses 5).
4. `dedup_rmsd_tol` and `dedup_energy_tol` — if the archive over-merges
   distinct basins or over-splits the same basin.
5. `max_step_scale`, `walk_trust_radius`, and `proposal_trust_radius` — these
   set the physical displacement envelope.  Slabs usually need tighter values
   than clusters.
6. `target_uphill_energy` — the bias hill height.  0.8 eV works for validated
   C60/PdO settings, but metal slabs should start lower (0.3-0.5 eV).
7. `max_steps_per_walk` — 8 is reasonable for C60; 4-5 is safer for slabs until
   displacement diagnostics show no layer buckling or cross-layer atom drift.
8. `proposal_relax_steps` — 10 works as a speed knob (~40% cheaper per
   trial) but has higher trajectory variance than 80.  Use 10 for parameter
   sweeps, 80 for production.

Leave bandit weights and lower-level direction-scoring modes at defaults until
you have per-step diagnostics showing a specific failure mode.  Most plateau
problems trace to walker escape capability, not selector or direction-scoring
details.

### Experimental Direction Diagnostics

The following direction controls are experimental or diagnostic only.
Production defaults remain unchanged, and these should not be promoted to
production knobs until they pass multi-seed validation.

- `direction_diagnostics_enabled=True` with
  `direction_diagnostics_path="runs/debug/direction_trace.jsonl"` writes a
  per-step trace of the selected direction kind, candidate count, curvature,
  and anchor cosine.
- `enable_anchor_candidate=True` is deprecated and retained only for
  configuration compatibility.  Raw anchor-as-candidate was withdrawn after C60
  smokes showed anchor-collapse and worse minima.  The walk anchor remains
  available as a prior for scoring, softening, and regularized Ritz synthesis.
- `choice_aligned_softening_enabled=True` applies only to LS-SSW runs using
  active-neighbor local softening.  If the chosen direction strongly disagrees
  with the anchor, active-neighbor softening is rebuilt from the chosen
  direction.  The default disagreement threshold is
  `choice_aligned_softening_cos_threshold=0.3`.
- `direction_synthesis_mode="regularized_ritz"` appends one `RITZ_REG`
  synthetic candidate built from the top `regularized_ritz_top_k` scored base
  candidates under `direction_selection_mode="discrete"`.  It reuses already
  computed candidate HVPs and remains experimental; default is
  `direction_synthesis_mode="none"`.
- `direction_type_ucb_enabled=True` enables an experimental, default-off UCB
  memory over direction kinds.  It is type-level (`MOMENTUM`, `BOND`,
  `RANDOM`, `RITZ_REG`, etc.), not per concrete direction identity, so it can
  help underused candidate classes compete without changing candidate
  generation.  It complements the planned finite-step probe diagnostics but
  does not replace them.  Tune with `direction_type_success_weight`,
  `direction_type_exploration_weight`, and `direction_type_ucb_window`.
- `direction_archive_enabled=True` enables the experimental, default-off
  `DirectionArchive`.  This is passive data collection for future `EVOLVED`
  concrete direction proposals; it is not an active selector and does not
  generate new candidates or change scoring.  Records are attributed per
  proposal, not smeared across every direction chosen in the trial.  Set
  `direction_archive_path` to write a per-run JSONL audit log for offline
  analysis; each `run()` resets that file, while the in-memory archive remains
  bounded by `direction_archive_max_records`.
- `random_direction_distribution="mass_weighted"` makes random Cartesian
  components scale as `1/sqrt(mass)`.  This is neutral after normalization for
  homonuclear systems such as C60, but gives lighter atoms larger random kicks
  in heterogeneous systems such as PdO.  Default remains
  `"unit_gaussian"`.
- `enable_bond_form_break_split=True` replaces dynamic legacy `BOND` proposals
  with explicit `BOND_FORM` and `BOND_BREAK` candidates.  Formation candidates
  use the window `adaptive_non_neighbor_threshold < distance <
  bond_formation_max_distance`; breaking candidates use `distance <
  bond_breaking_max_distance`.  Manually configured `bond_pairs` remain legacy
  `BOND` candidates for compatibility.  Defaults keep the legacy `BOND` path
  unchanged.
- `step_length_mode="per_atom_rms"` is the production default execution-step
  control.  It chooses `sigma` from the selected direction so the predicted
  per-atom RMS displacement is near `target_step_rms` and capped by
  `max_step_rms`.  The trust-region `sigma_scale` still shrinks or expands
  that target RMS.  Set `step_length_mode="curvature_adaptive"` only for
  legacy comparisons with the earlier curvature-based `sigma` path.  Use the
  summary fields
  `step_displacement_rms_*` and `step_displacement_max_atom_*` to audit whether
  slab or heterogeneous systems are receiving physically reasonable kicks.
  Because bias strength is still computed from `sigma^2 * curvature`, also
  check `bias_weight_max` and `bias_zero_weight_fraction`; repeated clipping
  means the RMS step mode is not yet safe for that system.

## Known Failure Modes

### Long Plateaus After Early Success

On C60, a 500-trial production run reached -508.88 eV by trial 163 and then
spent 337 trials (67% of budget) with no further improvement.  The walker was
still escaping seeds successfully (87.5% of walks reached a different basin),
but all destinations were above the global best.  This is a **walker escape
ceiling**: the archive has saturated the current funnel, and the walker's
direction candidates (momentum + bond + random) cannot find the funnel exit.

Mitigations under investigation: fewer walk steps to reduce overshooting,
better escape-direction candidates, and per-step diagnostics that identify
when a walk has already crossed into a lower-energy basin.

### Metropolis Chain vs UCB Archive

A Metropolis chain selector (T=0.26 eV) was tested against the UCB archive
selector.  At 40 trials the chain was clearly worse (-498.6 vs -502.9); at
100 trials it closed the best-energy gap (-505.8 vs -505.8) but with much
higher duplicate rate (0.535 vs 0.144).  The archive-based UCB selector
remains the production default.

### Direction Scoring: Adaptive Sigma

The default adaptive sigma mode uses per-candidate step scales in direction
scoring. Lower-level scoring modes should be treated as experimental until
validated by per-step diagnostics and multi-seed runs.

### Single-Seed Variance

C60 seed variance across 5 seeds with identical parameters spans ~6 eV
(-499.8 to -505.7 at 40 trials).  A single-seed result is not a reliable
indicator of method quality.  Always run at least 3 seeds for baselines.
