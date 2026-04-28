# pamssw

`pamssw` is a small V1 implementation of low-parameter `SSW` and `LS-SSW` for global optimization on fixed-cell atomistic systems.

## What is included

- Unified `SSW` / `LS-SSW` walker kernel
- Analytic and ASE calculator backends
- Proposal-only Gaussian bias terms
- Proposal-only pairwise local softening for `LS-SSW`
- Archive-based basin discovery
- Minimal YAML-driven CLI

## What is not included

- `DESW`
- Transition-state refinement
- IRC or fake-IRC
- Variable-cell search
- Canonical sampling or MH correction

## Python API

```python
from pamssw import LSSSWConfig, SSWConfig, State, run_ls_ssw, run_ssw
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D

state = State(numbers=[1], positions=[[-1.0, 0.0, 0.0]])
result = run_ssw(
    state,
    AnalyticCalculator(DoubleWell2D()),
    SSWConfig(max_trials=8, max_steps_per_walk=8, target_uphill_energy=0.8, rng_seed=7),
)
```

LS-SSW local softening defaults to automatic neighbor pairs:

```python
from pamssw import LSSSWConfig

config = LSSSWConfig(
    local_softening_mode="neighbor_auto",
    local_softening_cutoff_scale=1.25,
    local_softening_strength=0.6,
)
```

`manual` mode remains available for legacy workflows that need exact pair control, but `neighbor_auto` is the default because LS-SSW should derive local softening pairs from each walk's seed structure. The current penalty remains a Gaussian well; Buckingham/adaptive-strength variants should be added later as pluggable ablations after the auto-pair baseline is established.

Optional LS-SSW ablation controls are available without changing defaults:

```python
config = LSSSWConfig(
    local_softening_penalty="buckingham_repulsive",
    local_softening_xi=0.5,
    local_softening_adaptive_strength=True,
)
```

## CLI

```bash
python -m pamssw run-ssw config.yaml
python -m pamssw run-ls-ssw config.yaml
```

Minimal config shape:

```yaml
state:
  numbers: [1]
  positions:
    - [-1.0, 0.0, 0.0]

calculator:
  kind: analytic
  potential: double_well_2d

search:
  max_trials: 8
  max_steps_per_walk: 8
  target_uphill_energy: 0.8
  rng_seed: 7

output: result.json
```

## Verification

```bash
pytest -q tests/unit tests/integration
```
