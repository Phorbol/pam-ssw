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
