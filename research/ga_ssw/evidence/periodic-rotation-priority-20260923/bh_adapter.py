"""Research-only ASE BasinHopping adapter with PAM Safe-total quenches."""
from __future__ import annotations

from io import StringIO

import numpy as np
from ase import units
from ase.optimize.basin import BasinHopping

from pamssw.standalone.surface import SurfaceCalculator, quench


def run_bh(atoms, surface, *, seed, config, steps, on_quench, logfile=None):
    """Run ASE BasinHopping while sharing the supplied surface accounting.

    ``on_quench(result, cumulative_surface_requests)`` is called after every
    returned quench, including a non-converged result.  Exceptions from the
    surface, budget, or callback propagate unchanged.  The returned tuple is
    ``(BasinHopping, logfile)``; pass a caller-owned logfile to preserve output
    when an exception is handled by the caller.
    """
    np.random.seed(seed)  # ASE's global RandomState; distinct from SSW default_rng.
    stream = logfile if logfile is not None else StringIO()
    results = []

    class SafeTotalOptimizer:
        def __init__(self, optimizable, *, logfile=None):
            self.optimizable = optimizable
            self.logfile = logfile
            self.result = None

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def run(self, fmax=None, steps=None):
            before = surface.requests
            result = quench(self.optimizable.atoms, surface, fmax=config.fmax,
                            steps=config.relax_steps, optimizer="safe-lbfgs-total",
                            lbfgs_memory=config.lbfgs_memory)
            if result.evaluation_requests != surface.requests - before:
                raise AssertionError("quench/surface request count mismatch")
            self.result = result
            results.append(result)
            on_quench(result, surface.requests)
            if not result.converged:
                raise RuntimeError("failed local Safe-total quench")
            self.optimizable.atoms.set_positions(result.atoms.positions)
            return True

    def factory(optimizable, logfile=None):
        return SafeTotalOptimizer(optimizable, logfile=logfile)

    atoms.calc = SurfaceCalculator(surface)
    basin = BasinHopping(atoms, temperature=config.temperature_K * units.kB,
                         dr=0.5, fmax=config.fmax, logfile=stream,
                         trajectory=None, optimizer=factory,
                         optimizer_logfile=None, local_minima_trajectory=None)
    basin.run(steps)
    return basin, stream
