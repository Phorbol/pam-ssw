from __future__ import annotations

from .config import LSSSWConfig, RelaxConfig, SSWConfig
from .relax import Relaxer
from .walker import SurfaceWalker


def relax_minimum(state, calculator, config: RelaxConfig) -> object:
    relaxer = Relaxer(calculator.evaluate_flat)
    return relaxer.relax(state, fmax=config.fmax, maxiter=config.maxiter)


def run_ssw(initial_state, calculator, config: SSWConfig):
    walker = SurfaceWalker(calculator=calculator, config=config, softening_enabled=False)
    return walker.run(initial_state)


def run_ls_ssw(initial_state, calculator, config: LSSSWConfig):
    walker = SurfaceWalker(calculator=calculator, config=config, softening_enabled=True)
    return walker.run(initial_state)
