# Independent ASE implementation — development status

This package implements search components in Python using ASE calculators for
the physical potential. It does not launch LASP or Java. It does not call the
existing PAM walker. Uploaded programs are external test oracles only.

**Executable nonperiodic paper-reference drivers now exist:** `run_ssw`,
`run_ls_ssw`, and `run_ga_ssw`. GA currently supports the fixed-internal-unit
TYPE3 branch. This is not full uploaded-release parity or all-system support.
The runtime search is independent; ASE calculators may themselves use compiled
MLIP, tight-binding, or DFT libraries/programs as physical energy oracles.

Implemented interfaces:

- `surface.ASESurface(calculator, force_consistent=False).evaluate(atoms)`:
  energy/forces; no ownership change to the input Atoms. Choose
  `force_consistent=True` if the calculator's forces differentiate its
  free_energy. Missing backend capabilities propagate as errors. Backend
  energy/force consistency is not inferred from the ASE interface.
- `surface.quench(atoms, surface, fmax=..., steps=...)`: fixed-cell true
  quench; inspect `converged`, `max_force`, and `surface` on the result.
  Supplying `terms` instead returns a modified-surface certificate.
- `gaussian.ProjectedGaussian` and `GaussianSum`: additive projected bias;
  `adjust_native_weight`: static finite-domain 87-degree height update.
- `direction.reference_soft_mode`: bounded finite-difference Ritz solver.
  This is explicitly NOT the native biased Broyden dimer rotation.
- `softening.FrozenBondSoftening`: paper exponential pair potential; explicit
  element-pair energy/neighbor tables, frozen neighbors and reference lengths.
  `LSResponseState` updates the next step from true pre-quench energy response.
- `ls_cycle.prepare_ls_step` / `finish_ls_step`: soft-only preparation and
  true-surface finishing, with explicit failures on unconverged stages.
- `legacy_descriptor`: independently implemented archived NNA/selection
  behaviors. Its known atom-order dependence is preserved and documented;
  it is not a general structural-equivalence certificate.
- `population.partition` / `rank_regions`: Java-derived projection grouping,
  parent capping and empirical scores. Regions are not kinetic funnels.
- `ga_operators`: molecular operations and geometric docking; see
  `research/ga_ssw/GA_OPERATORS.md` for supported TYPE3 branches and provenance.

Minimal currently runnable physical-surface example:

```python
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface, quench

atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
surface = ASESurface(EMT())
result = quench(atoms, surface, fmax=1e-3, steps=100)
assert result.converged  # force stationarity only, not a Hessian certificate
```

Complete ordinary SSW example using the tested Cu13/EMT configuration:

```python
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, run_ssw

config = SSWConfig(width=.2, rotation_bias=10., max_gaussians=1,
    temperature_K=300., fmax=.001, relax_steps=100,
    fd_step=.001, rotation_hvp=40, rotation_tol=.02)
result = run_ssw(Icosahedron('Cu', 2), ASESurface(EMT()),
    steps=1, config=config, rng=np.random.default_rng(9))
print(result.status, [step.status for step in result.records])
```

These are short integration settings, not recommended optimal search defaults.
Replace Atoms and Calculator with the intended physical system/backend.
`run_ls_ssw` additionally requires `ls=LSSettings(...)`; the C60 script shows
explicit paper-derived parameters and true energy-response recording.
`run_ga_ssw` additionally takes initial structures, groups, frozen descriptor
references and `PaperGAConfig`; the water script provides a complete tested
invocation. Install the package from this checkout with `python -m pip install .`.

EMT is an example backend, not a requirement. Each concurrent walker must
own a calculator and, for file calculators, its own calculation directory.
Common surface/quench currently rejects constraints. Gaussian positions must
remain unwrapped through an escape; LS uses MIC on its frozen fixed cell.
No joint variable-cell search or stress interface is claimed by these modules.

`paper_reference` implements SSW2013 steps 1-8, BP-CBD2012 forward-force
height and the LS lifecycle. It explicitly substitutes one-sided Ritz rotation
for native Broyden and uses ASE LBFGS. Paper direction sampling combines a
Maxwell global direction with a >3 Angstrom atom-pair move; an explicit global
variant supports small systems lacking that pair, without a silent fallback.
`paper_ga` connects quick exploration, TYPE3 generation/short searches, and
ranked-region fine searches. Budget choices are explicit; uploaded-release
hidden multipliers/carry rules are not silently reproduced.

Remaining algorithm work: native Broyden/stopping/MC execution parity,
variable-internal-unit TYPE3 branches, TYPE0 and other original system types,
and periodic/RC/VC extensions. Recovered release bugs and paper-level
algorithm behavior must remain distinguishable.
The Cu13/EMT probe in `research/ga_ssw/evidence/independent-cu13-surface/`
checks surface/direction integration only. It is not an SSW search benchmark.
