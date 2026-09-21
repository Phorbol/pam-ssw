"""Research adapter for the experimental public bias-quench hook."""

from dataclasses import asdict, replace

import numpy as np
from ase.optimize import BFGS

from pamssw.standalone.surface import QuenchResult
from research.ga_ssw.native_stage_quench import NativeStageSnapshot, stage_aware_quench


_CONTEXT_KEYS = {
    "current", "current_energy", "best_energy", "outer_index",
    "gaussian_index", "center", "soft_terms", "max_gaussians",
}


def _outcome(relaxed, *, stage_stopped, release_all, diagnostics):
    from pamssw.standalone.paper_reference import BiasStageQuenchOutcome
    return BiasStageQuenchOutcome(relaxed=relaxed,
                                  stage_stopped=bool(stage_stopped),
                                  release_all=bool(release_all),
                                  diagnostics=diagnostics)


class StatefulNativeStageAdapter:
    """Callable matching ``run_ssw``'s experimental adapter contract."""

    def __init__(self, *, predicate_kwargs, stop_on, energy_reference,
                 gm_reference):
        if energy_reference not in ("current", "best"):
            raise ValueError("energy_reference must be 'current' or 'best'")
        if gm_reference not in ("current", "best"):
            raise ValueError("gm_reference must be 'current' or 'best'")
        if stop_on not in ("known_stop", "allstop"):
            raise ValueError("stop_on must be 'known_stop' or 'allstop'")
        self.predicate_kwargs = dict(predicate_kwargs)
        self.stop_on = stop_on
        self.energy_reference = energy_reference
        self.gm_reference = gm_reference
        self._outer_index = None
        self._maxe_height = 0.0

    def __call__(self, atoms, surface, *, fmax, steps, terms, optimizer,
                 frame, lbfgs_memory, context):
        if frame is not None:
            raise NotImplementedError("stage-aware adapter supports all-mobile Cartesian coordinates only")
        adapter_before = surface.requests
        if not isinstance(context, dict) or set(context) != _CONTEXT_KEYS:
            missing = sorted(_CONTEXT_KEYS.difference(context or {}))
            extra = sorted(set(context or {}).difference(_CONTEXT_KEYS))
            raise ValueError(f"context keys mismatch; missing={missing}, extra={extra}")
        if (self._outer_index != context["outer_index"]
                or int(context["gaussian_index"]) == 0):
            self._outer_index = context["outer_index"]
            self._maxe_height = 0.0
        all_terms = tuple(terms)
        soft_terms = tuple(context["soft_terms"])
        if len(soft_terms) > len(all_terms) or any(
                left is not right for left, right in zip(soft_terms, all_terms)):
            raise ValueError("context soft_terms must be the prefix of terms")
        gaussian_terms = all_terms[len(soft_terms):]
        center = np.asarray(context["center"], dtype=float)
        if center.shape != atoms.positions.shape or not np.isfinite(center).all():
            raise ValueError("context center must match finite atom positions")
        center_atoms = atoms.copy()
        center_atoms.set_positions(center)
        center_energy, _ = surface.evaluate(center_atoms)
        for term in soft_terms:
            de, _ = term.evaluate(center_atoms)
            center_energy += float(de)
        values = dict(self.predicate_kwargs)
        values.update(
            initial_energy=float(context[f"{self.energy_reference}_energy"]),
            saved_gaussian_energy=float(center_energy),
            maxe_height=self._maxe_height,
            ng=int(context["gaussian_index"]) + 1,
            para_ng=int(context["max_gaussians"]),
        )
        state = {}
        result = stage_aware_quench(
            atoms, surface, fmax=fmax, steps=steps,
            predicate_kwargs=values,
            gm_reference_energy=float(context[f"{self.gm_reference}_energy"]),
            stop_on=self.stop_on, base_terms=soft_terms,
            gaussian_terms=gaussian_terms,
            optimizer=optimizer if optimizer is not None else BFGS,
            frame=frame, lbfgs_memory=lbfgs_memory, monitor_state=state,
        )
        self._maxe_height = float(state.get("maxe_height", self._maxe_height))
        stage_requests = int(surface.requests - adapter_before)
        diagnostics = {
            "outer_index": context["outer_index"],
            "gaussian_index": context["gaussian_index"],
            "energy_reference": self.energy_reference,
            "gm_reference": self.gm_reference,
            "center_base_energy": float(center_energy),
            "center_reference_requests": 1,
            "local_consumptions": state.get("local_consumptions"),
            "maxe_height": self._maxe_height,
        }
        if isinstance(result, NativeStageSnapshot):
            # The adapter-local result includes the center reference request.
            local_requests = stage_requests
            relaxed = QuenchResult(
                result.atoms.copy(), result.modified_energy,
                result.modified_max_force, result.force_qualified,
                None, local_requests, "modified", None,
            )
            diagnostics.update(
                surface_requests=surface.requests,
                modified_max_force=result.modified_max_force,
                force_qualified=result.force_qualified,
                optimizer_converged=False, optimizer_steps=None,
                decision=asdict(result.decision),
            )
            return _outcome(relaxed, stage_stopped=True,
                            release_all=bool(result.decision.allstop),
                            diagnostics=diagnostics)
        result = replace(result, evaluation_requests=stage_requests)
        diagnostics.update(
            surface_requests=surface.requests,
            optimizer_converged=bool(result.converged),
            optimizer_steps=result.optimizer_steps,
        )
        return _outcome(result, stage_stopped=False, release_all=False,
                        diagnostics=diagnostics)


def make_stage_adapter(**kwargs):
    return StatefulNativeStageAdapter(**kwargs)


__all__ = ["StatefulNativeStageAdapter", "make_stage_adapter"]
