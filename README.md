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

## Install

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e '.[dev]'
pytest -q tests/unit tests/integration
```

## Recommended Production Pattern

For real atomistic work, start with `run_ls_ssw` and these defaults:

- `local_softening_mode="neighbor_auto"` so pairs are rebuilt from each seed.
- `proposal_optimizer="ase-fire"` for robust short proposal relaxations.
- `quench_optimizer="ase-fire"` or `ase-lbfgs` for ASE-backed systems.
- `accepted_structures_log` and `accepted_structures_dir` enabled.
- `write_proposal_minima=True` when diagnosing duplicates or failed proposals.
- `write_relaxation_trajectories=False` for routine runs; enable it only for
  debugging because it creates many files.

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
    proposal_optimizer="ase-fire",
    quench_optimizer="ase-fire",
    local_softening_mode="neighbor_auto",
    local_softening_penalty="buckingham_repulsive",
    local_softening_xi=0.3,
    local_softening_cutoff=2.0,
    accepted_structures_dir="runs/slab/accepted_minima",
    accepted_structures_log="runs/slab/accepted_structures.jsonl",
)
```

`ASECalculator` preserves `cell` and `pbc` when it builds ASE `Atoms`. Local
softening and pair-distance fingerprints use minimum-image distances on periodic
axes. The search remains fixed-cell: do not expect lattice-vector optimization.

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

Start with only these:

- `max_trials`: increase for deeper search.
- `rng_seed`: change for independent runs.
- `max_steps_per_walk`: increase for harder basin exits.
- `proposal_relax_steps`: increase when proposal relaxations under-converge.
- `quench_fmax`: tighten for final minima.
- `dedup_rmsd_tol` and `dedup_energy_tol`: adjust if the archive over-merges or
  over-splits basins.
- `local_softening_strength`: adjust LS-SSW push strength.

Leave lower-level direction-scoring and bandit weights at defaults until you
have accepted/proposal minima and trajectory diagnostics showing a specific
failure mode.

## More Details

- Parameter reference: `docs/developer-parameters.md`
- Theory and design notes: `docs/theoretical-analysis.md`
- Direction-oracle notes: `docs/direction-oracle-theory-and-plan.md`
