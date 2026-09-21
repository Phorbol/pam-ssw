"""Shared numerical all-DOF quench, without a search or MC lifecycle."""
from dataclasses import dataclass
import numpy as np
from .vc_geometry import SymmetricLogStrainChart
from .generalized_numerics import safe_lbfgs


def relax_cell_coordinates(chart, q, surface, *, pressure, fmax, stress_tol,
                           max_step, maxiter, lbfgs_memory=None):
    """Safe-total on E+pV with accepted-state physical stopping criteria."""
    convergence = {}
    def norm(v):
        return max(float(np.linalg.norm(v[:-6].reshape(-1,3),axis=1).max()),
                   float(np.linalg.norm(v[-6:])))
    def evaluate(x):
        ev = chart.evaluate(x, surface.evaluate, pressure=pressure)
        convergence[x.tobytes()] = max(
            float(np.linalg.norm(ev.forces,axis=1).max())/fmax,
            float(np.abs(ev.stress+pressure*np.eye(3)).max())/stress_tol)
        return ev.objective, chart.project(ev.gradient)
    return safe_lbfgs(q, evaluate, gradient_norm=norm, step_norm=norm,
        convergence_norm=lambda x,g: convergence[x.tobytes()], gtol=1.,
        max_step=max_step, maxiter=maxiter, lbfgs_memory=lbfgs_memory)


@dataclass
class CellQuenchResult:
    evaluation: object
    optimizer: object
    certificate: dict
    requests: int

    @property
    def converged(self):
        return self.optimizer.converged and self.certificate.get('certified',False)


def cell_quench(atoms, surface, *, strain_length, pressure, fmax, stress_tol,
                max_step, maxiter, lbfgs_memory=None):
    """One pure all-DOF quench and fresh certificate; no proposal or selection."""
    for name,value in [('fmax',fmax),('stress_tol',stress_tol)]:
        if not np.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive')
    before=surface.requests
    chart=SymmetricLogStrainChart(atoms,strain_length=strain_length)
    result=relax_cell_coordinates(chart,chart.pack(atoms),surface,
        pressure=pressure,fmax=fmax,stress_tol=stress_tol,max_step=max_step,maxiter=maxiter,
        lbfgs_memory=lbfgs_memory)
    if result.energy is None:
        return CellQuenchResult(None,result,dict(certified=False),surface.requests-before)
    try:
        ev=chart.evaluate(result.q,surface.evaluate,pressure=pressure)
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        return CellQuenchResult(None,result,dict(certified=False,error=str(error)),surface.requests-before)
    force=float(np.linalg.norm(ev.forces,axis=1).max())
    stress=float(np.abs(ev.stress+pressure*np.eye(3)).max())
    certificate=dict(fmax=force,stress_max=stress,certified=force<=fmax and stress<=stress_tol)
    return CellQuenchResult(ev,result,certificate,surface.requests-before)
