"""Research-only stage monitor for the fixed-cell quench boundary.

This adapter composes the existing :func:`pamssw.standalone.surface.quench`
and its Safe-total/ASE optimizers.  It does not implement an optimizer or a
native lifecycle.  A monitor runs after one physical ``E/F`` request and after
the explicitly supplied base (for example LS) and Gaussian terms have been
formed.  A private exception stops the optimizer at that evaluated snapshot.

The returned point is a stage snapshot, not an accepted point and not a true
minimum.  ``optimizer_steps`` is intentionally ``None`` because an optimizer
may have consumed a trial internally before the monitor fired.  Callers must
perform the ordinary true-surface certificate separately.
"""

from dataclasses import dataclass

import numpy as np
from ase.optimize import BFGS

from pamssw.standalone.surface import ASESurface, quench
from research.ga_ssw.native_stage_predicate import (
    NativeStagePredicate,
    closed_native_stage_predicate,
)


@dataclass(frozen=True)
class NativeStageSnapshot:
    atoms: object
    physical_energy: float
    physical_forces: np.ndarray
    base_energy: float
    base_forces: np.ndarray
    gaussian_energy: float
    gaussian_forces: np.ndarray
    modified_energy: float
    modified_forces: np.ndarray
    modified_max_force: float
    force_qualified: bool
    decision: NativeStagePredicate
    evaluation_requests: int
    local_consumptions: int
    optimizer_steps: None = None
    converged: bool = False
    stage_stop_reason: str = "native_stage_predicate"


class _NativeStageStop(RuntimeError):
    def __init__(self, snapshot):
        super().__init__("native stage predicate requested a stage stop")
        self.snapshot = snapshot


class _MonitoredSurface:
    """Expose a base surface while retaining its request accounting."""

    def __init__(self, physical, base_terms):
        if not isinstance(physical, ASESurface):
            raise TypeError("physical must be an ASESurface")
        self.physical = physical
        self.base_terms = tuple(base_terms)
        self._last = None

    @property
    def requests(self):
        return self.physical.requests

    def evaluate(self, atoms):
        energy, forces = self.physical.evaluate(atoms)
        forces = np.asarray(forces, dtype=float)
        base_energy = float(energy)
        base_forces = forces.copy()
        for term in self.base_terms:
            de, df = term.evaluate(atoms)
            df = np.asarray(df, dtype=float)
            if df.shape != forces.shape or not np.isfinite(df).all() or not np.isfinite(de):
                raise ValueError("base term returned invalid energy or forces")
            base_energy += float(de)
            base_forces += df
        # SurfaceCalculator adds every Gaussian term in-place to the force
        # array returned below.  Keep an independent base snapshot so the
        # stage monitor does not observe (and add) those terms a second time.
        self._last = (atoms.copy(), float(energy), forces.copy(),
                      base_energy, base_forces.copy())
        return base_energy, base_forces


class _StageMonitor:
    """Last additive term; it returns zero and may stop after one full E/F."""

    def __init__(self, owner, gaussian_terms, predicate_kwargs, stop_on,
                 *, fmax, gm_reference_energy):
        self.owner = owner
        self.gaussian_terms = tuple(gaussian_terms)
        self.predicate_kwargs = dict(predicate_kwargs)
        self.stop_on = stop_on
        self.fmax = float(fmax)
        self.gm_reference_energy = float(gm_reference_energy)
        self.local_consumptions = 0
        self.maxe_height = float(predicate_kwargs["maxe_height"])

    def evaluate(self, atoms):
        cached = self.owner._last
        if (cached is None or not np.array_equal(cached[0].positions, atoms.positions)
                or not np.array_equal(cached[0].cell.array, atoms.cell.array)
                or not np.array_equal(cached[0].pbc, atoms.pbc)):
            raise RuntimeError("stage monitor lost the matching physical snapshot")
        source, physical_energy, physical_forces, base_energy, base_forces = cached
        gaussian_energy = 0.0
        gaussian_forces = np.zeros_like(base_forces)
        for term in self.gaussian_terms:
            de, df = term.evaluate(atoms)
            df = np.asarray(df, dtype=float)
            if df.shape != base_forces.shape or not np.isfinite(df).all() or not np.isfinite(de):
                raise ValueError("Gaussian term returned invalid energy or forces")
            gaussian_energy += float(de)
            gaussian_forces += df
        modified_energy = base_energy + gaussian_energy
        modified_forces = base_forces + gaussian_forces
        self.local_consumptions += 1
        values = dict(self.predicate_kwargs)
        values["max_force"] = float(np.max(np.abs(modified_forces)))
        values["base_energy"] = float(base_energy)
        self.maxe_height = max(self.maxe_height,
                               base_energy - float(values["initial_energy"]))
        values["maxe_height"] = self.maxe_height
        values["maxe_height_gm"] = base_energy - self.gm_reference_energy
        # The stored native counter is incremented after each dispatch; the
        # first complete consumption therefore observes start + 1.
        values["climbstep"] = (int(values["counter_start"])
                                + self.local_consumptions)
        del values["counter_start"]
        decision = closed_native_stage_predicate(**values)
        trigger = decision.known_stop if self.stop_on == "known_stop" else decision.allstop
        if self.stop_on not in ("known_stop", "allstop"):
            raise ValueError("stop_on must be 'known_stop' or 'allstop'")
        if trigger:
            snapshot = NativeStageSnapshot(
                source, float(physical_energy), physical_forces.copy(),
                float(base_energy), base_forces.copy(), float(gaussian_energy),
                gaussian_forces.copy(), float(modified_energy),
                modified_forces.copy(),
                float(np.max(np.linalg.norm(modified_forces, axis=1))),
                float(np.max(np.linalg.norm(modified_forces, axis=1))) <= self.fmax,
                decision, self.owner.requests, self.local_consumptions,
            )
            raise _NativeStageStop(snapshot)
        return 0.0, np.zeros_like(base_forces)


def stage_aware_quench(
    atoms,
    surface,
    *,
    fmax,
    steps,
    predicate_kwargs,
    gm_reference_energy,
    stop_on,
    base_terms=(),
    gaussian_terms=(),
    optimizer=BFGS,
    frame=None,
    lbfgs_memory=None,
    monitor_state=None,
):
    """Run an existing quench and stop only at an evaluated stage snapshot.

    ``predicate_kwargs`` must contain every native scalar input required by
    ``closed_native_stage_predicate`` except measured ``max_force``,
    ``base_energy``, derived ``maxe_height_gm`` and the derived ``climbstep``.
    ``counter_start`` and ``gm_reference_energy`` are explicit caller inputs;
    there are no hidden counter or GM defaults. ``base_terms`` are evaluated once as part of the
    objective and may include LS; ``gaussian_terms`` are evaluated by the
    existing quench calculator and once more by the monitor without a physical
    calculator request.  The monitor is deliberately last so its zero term
    does not alter the objective.
    """
    if not isinstance(predicate_kwargs, dict):
        raise TypeError("predicate_kwargs must be an explicit dict")
    if frame is not None:
        raise NotImplementedError("stage monitor supports only all-mobile Cartesian coordinates")
    if not np.isfinite(gm_reference_energy):
        raise ValueError("gm_reference_energy must be finite")
    required = {
        "climb_stopf", "initial_energy", "saved_gaussian_energy",
        "maxe_height", "e_maxlimit", "f_maxlimit",
        "e_maxlimit_gm", "para_ng", "ng", "ngaus_relax",
        "ngaus_relax_ini", "multi_pes", "counter_start",
    }
    missing = sorted(required.difference(predicate_kwargs))
    if missing:
        raise ValueError("predicate_kwargs missing: " + ", ".join(missing))
    if "max_force" in predicate_kwargs or "base_energy" in predicate_kwargs:
        raise ValueError("max_force/base_energy are measured by the monitor")
    if stop_on not in ("known_stop", "allstop"):
        raise ValueError("stop_on must be 'known_stop' or 'allstop'")
    if "maxe_height_gm" in predicate_kwargs:
        raise ValueError("maxe_height_gm is derived from gm_reference_energy")
    monitored = _MonitoredSurface(surface, base_terms)
    monitor = _StageMonitor(
        monitored, gaussian_terms, predicate_kwargs, stop_on,
        fmax=fmax, gm_reference_energy=gm_reference_energy,
    )
    def publish_state():
        if monitor_state is not None:
            monitor_state.update(
                maxe_height=monitor.maxe_height,
                local_consumptions=monitor.local_consumptions,
            )
    try:
        result = quench(atoms, monitored, fmax=fmax, steps=steps,
                        terms=tuple(gaussian_terms) + (monitor,), optimizer=optimizer,
                        frame=frame, lbfgs_memory=lbfgs_memory)
    except _NativeStageStop as stop:
        publish_state()
        snapshot = stop.snapshot
        return snapshot
    except BaseException:
        publish_state()
        raise
    publish_state()
    return result


__all__ = ["NativeStageSnapshot", "stage_aware_quench"]
