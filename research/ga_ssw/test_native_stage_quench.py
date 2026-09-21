"""Small no-PES checks for the research stage adapter."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from pamssw.standalone.surface import ASESurface, quench
from pamssw.standalone.gaussian import ProjectedGaussian
from research.ga_ssw.native_stage_quench import NativeStageSnapshot, stage_aware_quench


class _Quadratic(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions.copy()
        self.results = {"energy": float(0.5 * np.sum(x * x)), "forces": -x}


class _ConstantTerm:
    def evaluate(self, atoms):
        return 0.25, np.zeros_like(atoms.positions)


class _ConstantForceTerm:
    def __init__(self, force):
        self.force = np.asarray(force, dtype=float)

    def evaluate(self, atoms):
        return 0.0, np.broadcast_to(self.force, atoms.positions.shape).copy()


def _inputs(**overrides):
    values = dict(
        climb_stopf=1.0, initial_energy=0.0, saved_gaussian_energy=0.0,
        maxe_height=0.0, e_maxlimit=99.0,
        f_maxlimit=99.0, e_maxlimit_gm=99.0, para_ng=3, ng=1,
        ngaus_relax=10, ngaus_relax_ini=10, multi_pes=False, counter_start=1,
    )
    values.update(overrides)
    return values


def test_snapshot_contains_base_and_gaussian_without_extra_physical_request():
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    surface = ASESurface(_Quadratic())
    result = stage_aware_quench(
        atoms, surface, fmax=0.01, steps=0, gm_reference_energy=0.0,
        predicate_kwargs=_inputs(),
        stop_on="known_stop", base_terms=(_ConstantTerm(),),
        gaussian_terms=(ProjectedGaussian(np.zeros((1, 3)),
                                          np.array([[1.0, 0.0, 0.0]]),
                                          1.0, 0.5),),
    )
    assert isinstance(result, NativeStageSnapshot)
    assert result.evaluation_requests == 1
    assert result.base_energy == pytest.approx(result.physical_energy + 0.25)
    assert result.gaussian_energy > 0.0
    assert np.array_equal(result.atoms.positions, atoms.positions)
    assert result.optimizer_steps is None and not result.converged


def test_allstop_is_distinct_from_force_stage_stop():
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    surface = ASESurface(_Quadratic())
    result = stage_aware_quench(
        atoms, surface, fmax=0.01, steps=0,
        gm_reference_energy=0.0,
        predicate_kwargs=_inputs(ng=2, para_ng=3, climb_stopf=0.01,
                                 e_maxlimit=0.01),
        stop_on="allstop",
    )
    assert isinstance(result, NativeStageSnapshot)
    assert result.decision.allstop
    assert not result.decision.force_stop


def test_finite_quench_budget_is_not_reported_as_stage_convergence():
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    surface = ASESurface(_Quadratic())
    result = stage_aware_quench(
        atoms, surface, fmax=0.01, steps=0,
        gm_reference_energy=0.0,
        predicate_kwargs=_inputs(climb_stopf=0.01, counter_start=12),
        stop_on="known_stop",
    )
    assert isinstance(result, NativeStageSnapshot)
    assert result.decision.step_over
    assert not result.converged


def test_failed_physical_request_remains_paid():
    class Failing(_Quadratic):
        def calculate(self, *args, **kwargs):
            raise RuntimeError("synthetic backend failure")

    surface = ASESurface(Failing())
    with pytest.raises(RuntimeError, match="synthetic"):
        stage_aware_quench(
            Atoms("H", positions=[[0.2, 0.0, 0.0]]), surface, fmax=0.01,
            steps=0, gm_reference_energy=0.0,
            predicate_kwargs=_inputs(), stop_on="known_stop",
        )
    assert surface.requests == 1


def test_safe_total_can_stop_after_multiple_consumptions():
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    surface = ASESurface(_Quadratic())
    result = stage_aware_quench(
        atoms, surface, fmax=1.0e-12, steps=5,
        gm_reference_energy=0.0,
        predicate_kwargs=_inputs(climb_stopf=1.0e-12, counter_start=1,
                                 ngaus_relax_ini=2),
        stop_on="known_stop", optimizer="safe-lbfgs-total",
    )
    assert isinstance(result, NativeStageSnapshot)
    assert result.local_consumptions == 2
    assert result.evaluation_requests == surface.requests
    assert result.decision.step_over
    assert result.modified_max_force >= 0.0


def test_second_backend_failure_is_paid():
    class FailsOnSecond(_Quadratic):
        calls = 0

        def calculate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic second-request failure")
            return super().calculate(*args, **kwargs)

    calculator = FailsOnSecond()
    surface = ASESurface(calculator)
    with pytest.raises(RuntimeError, match="second-request"):
        stage_aware_quench(
            Atoms("H", positions=[[0.2, 0.0, 0.0]]), surface, fmax=1.0e-12,
            steps=1, gm_reference_energy=0.0,
            predicate_kwargs=_inputs(climb_stopf=1.0e-12, counter_start=1),
            stop_on="known_stop",
        )
    assert surface.requests == 2


def test_no_trigger_matches_existing_quench_endpoint():
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    staged_surface = ASESurface(_Quadratic())
    plain_surface = ASESurface(_Quadratic())
    kwargs = _inputs(climb_stopf=-1.0, counter_start=1)
    staged = stage_aware_quench(
        atoms, staged_surface, fmax=0.01, steps=1, gm_reference_energy=0.0,
        predicate_kwargs=kwargs, stop_on="known_stop",
    )
    plain = quench(atoms, plain_surface, fmax=0.01, steps=1)
    assert isinstance(staged, type(plain))
    assert staged.evaluation_requests == plain.evaluation_requests
    assert staged.energy == pytest.approx(plain.energy)
    assert np.array_equal(staged.atoms.positions, plain.atoms.positions)


def test_gaussian_force_is_added_once_to_cached_base_and_modified_forces():
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    result = stage_aware_quench(
        atoms, ASESurface(_Quadratic()), fmax=0.01, steps=0,
        gm_reference_energy=0.0,
        predicate_kwargs=_inputs(climb_stopf=0.35), stop_on="known_stop",
        gaussian_terms=(_ConstantForceTerm([0.3, 0.0, 0.0]),),
    )
    assert result.base_forces[0, 0] == pytest.approx(-0.2)
    assert result.modified_forces[0, 0] == pytest.approx(0.1)


def test_force_gate_uses_single_gaussian_contribution():
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    result = stage_aware_quench(
        atoms, ASESurface(_Quadratic()), fmax=0.01, steps=0,
        gm_reference_energy=0.0,
        predicate_kwargs=_inputs(climb_stopf=0.35), stop_on="known_stop",
        gaussian_terms=(_ConstantForceTerm([0.3, 0.0, 0.0]),),
    )
    assert result.decision.force_stop
