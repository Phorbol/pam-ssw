"""Research-only VC-SSW runner using mature L-BFGS adapters.

This module temporarily redirects the two Safe-total call sites and restores
them even when a proposal fails.  It does not alter production defaults.
"""
from contextlib import contextmanager
import numpy as np

from pamssw.standalone import vc_reference, cell_relax
from .lbfgs_baselines import scipy_lbfgsb, ase_lbfgs_linesearch


class _CompatRelax:
    """Expose only fields consumed by VC-SSW; no fabricated secant counts."""
    def __init__(self, result):
        self._result = result
        for name in ("q", "energy", "gradient", "status", "steps", "requests",
                     "trace", "error", "converged"):
            setattr(self, name, getattr(result, name))
        self.rejected_trials = result.metadata["distinct_rejected_trials"]


@contextmanager
def temporary_lbfgs_baseline(kind, records=None, *, native_gradient_tol=None):
    """Redirect VC and cell quench Safe-total calls for one controlled run."""
    if kind == "scipy":
        adapter = scipy_lbfgsb
    elif kind == "ase":
        adapter = ase_lbfgs_linesearch
    else:
        raise ValueError("kind must be scipy or ase")
    old_vc, old_cell = vc_reference.safe_lbfgs, cell_relax.safe_lbfgs

    def bridge_for(source):
      def bridge(*args, **kwargs):
        call = {"kind": kind, "max_step": kwargs.get("max_step"),
                "requested_maxiter": kwargs.get("maxiter"), "source": source,
                "common_gtol": kwargs.get("gtol")}
        kwargs.pop("max_step", None)
        conversion = None
        if source == "vc_reference" and kwargs.get("convergence_norm") is None:
            common = float(kwargs["gtol"])
            conversion = {"common_gtol": common, "kind": kind,
                          "sufficient": True, "necessary": False}
            if kind == "scipy":
                kwargs["native_gtol"] = common / np.sqrt(6.0)
            else:
                kwargs["native_fmax"] = common / np.sqrt(2.0)
        if kwargs.get("lbfgs_memory") is None:
            kwargs["lbfgs_memory"] = None
        if native_gradient_tol is not None:
            if conversion is None:
                kwargs["native_gtol"] = native_gradient_tol
        if kind == "ase":
            if conversion is None:
                kwargs["native_fmax"] = native_gradient_tol or kwargs.get("gtol")
        result = adapter(*args, **kwargs)
        call.update(status=result.status, requests=result.requests,
                    steps=result.steps, native_success=result.native_success,
                    certificate=result.certificate, metadata=result.metadata,
                    norm_conversion=conversion)
        if records is not None:
            # Observe accepted points without changing native termination.
            # Index zero is the initial point; a transient pass does not
            # override the terminal status or its physical certificate.
            convergence_norm = kwargs.get("convergence_norm")
            if convergence_norm is None:
                convergence_norm = kwargs["gradient_norm"]
                criterion = lambda point: convergence_norm(point["gradient"].copy())
            else:
                criterion = lambda point: convergence_norm(
                    point["q"].copy(), point["gradient"].copy())
            first_passage = None
            for accepted_index, point in enumerate(result.accepted_trace):
                value = float(criterion(point))
                if value <= float(kwargs["gtol"]):
                    first_passage = (accepted_index, int(point["requests"]), value)
                    break
            call.update(
                first_common_qualified_accepted_index=(
                    None if first_passage is None else first_passage[0]),
                first_common_qualified_request=(
                    None if first_passage is None else first_passage[1]),
                first_common_qualified_criterion=(
                    None if first_passage is None else first_passage[2]),
                requests_after_first_common_qualified=(
                    None if first_passage is None else result.requests - first_passage[1]))
            records.append(call)
        return _CompatRelax(result)
      return bridge

    vc_reference.safe_lbfgs = bridge_for("vc_reference")
    cell_relax.safe_lbfgs = bridge_for("cell_relax")
    try:
        yield records
    finally:
        vc_reference.safe_lbfgs = old_vc
        cell_relax.safe_lbfgs = old_cell


def run_vc_baseline(atoms, surface, *, steps, config, rng, kind,
                    fresh_surface=None, **kwargs):
    """Run one VC-SSW workflow and return the run plus phase accounting."""
    calls = []
    before = surface.requests
    with temporary_lbfgs_baseline(kind, calls,
                                  native_gradient_tol=config.gradient_tol):
        result = vc_reference.run_vc_ssw(atoms, surface, steps=steps,
                                         config=config, rng=rng, **kwargs)
    fresh_surface = surface if fresh_surface is None else fresh_surface
    after_run = surface.requests
    fresh_before = fresh_surface.requests
    fresh = []
    for minimum in result.minima:
        energy, forces, stress = fresh_surface.evaluate(minimum.atoms)
        stress_residual = stress + config.pressure * np.eye(3)
        fresh.append({"energy": float(energy),
                      "fmax": float(np.linalg.norm(forces, axis=1).max()),
                      "stress_max": float(np.abs(stress_residual).max()),
                      "certified": float(np.linalg.norm(forces, axis=1).max()) <= config.fmax and
                                   float(np.abs(stress_residual).max()) <= config.stress_tol})
    phases = []
    cell_seen = 0
    for call in calls:
        if call["source"] == "cell_relax":
            phase = "initial_truequench" if cell_seen == 0 else "truequench"
            cell_seen += 1
        else:
            phase = "biased"
        phases.append({"phase": phase, **call})
    return {"kind": kind, "result": result, "adapter_calls": calls,
            "phases": phases, "fresh": fresh,
            "efs_requests": after_run - before,
            "fresh_requests": fresh_surface.requests - fresh_before,
            "total_requests": after_run - before + fresh_surface.requests - fresh_before}
