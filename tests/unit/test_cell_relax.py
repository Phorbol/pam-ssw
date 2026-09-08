import numpy as np
import pytest
from ase import units
from ase.filters import FrechetCellFilter
from pamssw.accounting import BudgetExceeded, EvalCounter
from pamssw.calculators import EnergyResult
from pamssw.state import State


class VolumeWell:
    def __init__(self, target=8.0, k=0.1):
        self.target, self.k = target, k
        self.cells = []

    def evaluate(self, state):
        self.cells.append(state.cell.copy())
        delta = np.linalg.det(state.cell) - self.target
        return EnergyResult(0.5 * self.k * delta**2,
                            np.zeros_like(state.positions),
                            np.eye(3) * self.k * delta)


def bulk(cell=None, fixed=False):
    return State([1], [[0.4, 0.6, 0.8]],
                 np.eye(3) * 2.2 if cell is None else cell,
                 (True, True, True), [fixed])


@pytest.mark.parametrize('optimizer', ['ase-lbfgs', 'ase-fire'])
@pytest.mark.parametrize('pressure', [0.0, 3.0])
def test_volume_optimum_pressure_and_fractional_fixed_atom(optimizer, pressure):
    from pamssw.cell_relax import CellRelaxer
    state = bulk(fixed=True)
    calc = VolumeWell()
    counted = EvalCounter(calc)
    result = CellRelaxer(counted, optimizer=optimizer, mode='volume_only',
                         pressure_gpa=pressure, stress_tol=1e-6).relax(
                             state, fmax=1e-5, maxiter=500)
    p = pressure * units.GPa
    assert result.volume == pytest.approx(8.0 - p / 0.1, abs=1e-5)
    assert result.energy == pytest.approx(result.potential_energy + p * result.volume)
    assert result.stress_norm <= 1e-6
    assert result.telemetry.converged
    assert result.gradient_norm == 0
    assert counted.force_evaluations == len(calc.cells) == result.telemetry.evaluator_calls
    assert not np.allclose(calc.cells[0], calc.cells[-1])
    np.testing.assert_allclose(result.state.positions @ np.linalg.inv(result.state.cell),
                               state.positions @ np.linalg.inv(state.cell), atol=1e-12)
    assert result.state.metadata['enthalpy'] == result.energy
    assert result.state.metadata['potential_energy'] == result.potential_energy
    assert result.state.metadata['volume'] == result.volume
    assert result.state.metadata['external_pressure_gpa'] == pressure


def test_stress_must_converge_even_with_zero_atomic_forces():
    from pamssw.cell_relax import CellRelaxer
    result = CellRelaxer(VolumeWell(), stress_tol=1e-8).relax(bulk(), fmax=1, maxiter=0)
    assert result.gradient_norm == 0
    assert result.stress_norm > 1e-8
    assert not result.telemetry.converged
    assert result.n_iter == 0


def test_budget_interrupt_propagates_without_double_counting():
    from pamssw.cell_relax import CellRelaxer
    calc = EvalCounter(VolumeWell(), max_force_evals=1)
    with pytest.raises(BudgetExceeded):
        CellRelaxer(calc).relax(bulk(), fmax=1e-5, maxiter=30)
    assert calc.force_evaluations == 1


@pytest.mark.parametrize('stress', [None, np.full((3, 3), np.nan), np.zeros(4)])
def test_missing_or_invalid_stress_fails(stress):
    from pamssw.cell_relax import CellRelaxer
    class BadStress:
        def evaluate(self, state):
            return EnergyResult(0, np.zeros_like(state.positions), stress)
    with pytest.raises(ValueError, match='stress'):
        CellRelaxer(BadStress()).relax(bulk(), fmax=1e-3, maxiter=1)


@pytest.mark.parametrize('cell', [np.zeros((3, 3)), np.eye(3) * np.nan, np.diag([-1, 2, 3])])
def test_invalid_cell_fails_before_evaluation(cell):
    from pamssw.cell_relax import CellRelaxer
    calc = VolumeWell()
    with pytest.raises(ValueError, match='cell'):
        CellRelaxer(calc).relax(bulk(cell), fmax=1e-3, maxiter=1)
    assert not calc.cells


def test_slab_fixes_vacuum_vector_and_rejects_pressure():
    from pamssw.cell_relax import CellRelaxer
    state = State([1], [[0.4, 0.6, 3]], np.diag([2.2, 2.2, 10]), (True, True, False))
    result = CellRelaxer(VolumeWell(target=40, k=0.01), mode='slab_xy', stress_tol=1e-6).relax(
        state, fmax=1e-4, maxiter=100)
    assert result.volume == pytest.approx(40, abs=1e-4)
    np.testing.assert_array_equal(result.state.cell[2], state.cell[2])
    assert result.state.positions[0, 2] == state.positions[0, 2]
    with pytest.raises(ValueError, match='pressure'):
        CellRelaxer(VolumeWell(), mode='slab_xy', pressure_gpa=1)
    with pytest.raises(ValueError, match='PBC|periodic'):
        CellRelaxer(VolumeWell()).relax(state, fmax=1e-3, maxiter=1)


def test_filter_gradient_at_nonzero_deformation_matches_energy_differences():
    from pamssw.cell_relax import CellRelaxer
    # Test all generalized coordinates independently of optimization/stopping.
    relaxer = CellRelaxer(VolumeWell(), pressure_gpa=2)
    atoms = relaxer._make_atoms(bulk())
    filt = FrechetCellFilter(atoms, scalar_pressure=2 * units.GPa)
    x = filt.get_positions()
    x[-3:] += np.array([[0.12, 0.04, 0], [0.04, -0.08, 0.03], [0, 0.03, 0.05]])
    filt.set_positions(x)
    force = filt.get_forces()
    for i, j in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i, j] += 1e-6
        xm[i, j] -= 1e-6
        filt.set_positions(xp)
        ep = filt.get_potential_energy()
        filt.set_positions(xm)
        em = filt.get_potential_energy()
        assert force[i, j] == pytest.approx(-(ep-em)/2e-6, abs=2e-7)
    filt.set_positions(x)


class AtomVolumeWell(VolumeWell):
    def evaluate(self, state):
        base = super().evaluate(state)
        # Cartesian harmonic atom term contributes an affine virial to stress.
        pos = state.positions
        volume = np.linalg.det(state.cell)
        return EnergyResult(base.energy + 0.5 * np.sum(pos**2), pos,
                            base.stress + pos.T @ pos / volume)


def test_joint_atomic_and_cell_gradient_at_nonzero_strain():
    from pamssw.cell_relax import CellRelaxer
    atoms = CellRelaxer(AtomVolumeWell(), pressure_gpa=4)._make_atoms(bulk())
    filt = FrechetCellFilter(atoms, scalar_pressure=4 * units.GPa)
    x = filt.get_positions()
    x[0] += [0.1, -0.05, 0.1]
    x[-3:] += np.array([[0.1, 0.04, -0.02], [0.04, -0.06, 0.03], [-0.02, 0.03, 0.04]])
    filt.set_positions(x)
    force = filt.get_forces()
    for i, j in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i, j] += 1e-6
        xm[i, j] -= 1e-6
        filt.set_positions(xp)
        ep = filt.get_potential_energy()
        filt.set_positions(xm)
        em = filt.get_potential_energy()
        assert force[i, j] == pytest.approx(-(ep-em)/2e-6, abs=2e-7)


def test_joint_quench_requires_atomic_force_convergence():
    from pamssw.cell_relax import CellRelaxer
    state = bulk(np.eye(3) * 2)
    incomplete = CellRelaxer(AtomVolumeWell(), stress_tol=1).relax(state, fmax=1e-5, maxiter=0)
    assert incomplete.stress_norm < 1
    assert incomplete.gradient_norm > 1e-5
    assert not incomplete.telemetry.converged
    result = CellRelaxer(AtomVolumeWell(), stress_tol=1e-7).relax(state, fmax=1e-6, maxiter=150)
    independent = AtomVolumeWell().evaluate(result.state)
    assert np.max(np.linalg.norm(independent.gradient, axis=1)) <= 1e-6
    assert np.max(np.abs(independent.stress)) <= 1e-7
    assert result.volume == pytest.approx(8, abs=1e-5)
    assert result.telemetry.converged


def test_periodic_lj_volume_quench_has_independent_force_and_stress_certificate():
    from ase.build import bulk as ase_bulk
    from ase.calculators.lj import LennardJones
    from pamssw.calculators import ASECalculator
    from pamssw.cell_relax import CellRelaxer
    atoms = ase_bulk('Ar', 'fcc', a=1.65, cubic=True)
    state = State(atoms.numbers, atoms.positions, atoms.cell.array, tuple(atoms.pbc))
    calc = ASECalculator(LennardJones())
    initial = calc.evaluate(state)
    result = CellRelaxer(calc, mode='volume_only', stress_tol=1e-5).relax(state, fmax=1e-5, maxiter=100)
    independent = ASECalculator(LennardJones()).evaluate(result.state)
    assert result.telemetry.converged
    assert independent.energy < initial.energy
    assert np.max(np.linalg.norm(independent.gradient, axis=1)) <= 1e-5
    assert abs(np.trace(independent.stress) / 3) <= 1e-5
    assert result.energy == pytest.approx(independent.energy)


def test_calculator_exception_preserved():
    from pamssw.cell_relax import CellRelaxer
    error = RuntimeError('backend exploded')
    class Broken:
        def evaluate(self, state):
            raise error
    with pytest.raises(RuntimeError) as caught:
        CellRelaxer(Broken()).relax(bulk(), fmax=1e-3, maxiter=1)
    assert caught.value is error


def test_slab_rejects_tilt_out_of_xy_before_evaluation():
    from pamssw.cell_relax import CellRelaxer
    cell = np.diag([2., 2., 10.])
    cell[0, 2] = 0.2
    calc = VolumeWell()
    with pytest.raises(ValueError, match='axis-aligned'):
        CellRelaxer(calc, mode='slab_xy').relax(bulk(cell), fmax=1e-3, maxiter=1)
    assert not calc.cells


def test_fixed_atom_retains_fractional_position_with_nonzero_raw_force():
    from pamssw.cell_relax import CellRelaxer
    state = State([1, 1], [[0.4, 0.6, 0.8], [0.6, 0.8, 0.4]],
                  np.eye(3)*2.2, (True, True, True), [True, False])
    result = CellRelaxer(AtomVolumeWell(), mode='volume_only', stress_tol=1e-7).relax(
        state, fmax=1e-6, maxiter=150)
    assert result.telemetry.converged
    raw = AtomVolumeWell().evaluate(result.state)
    assert np.linalg.norm(raw.gradient[0]) > 0.1
    assert np.linalg.norm(raw.gradient[1]) <= 1e-6
    np.testing.assert_allclose(result.state.positions[0] @ np.linalg.inv(result.state.cell),
                               state.positions[0] @ np.linalg.inv(state.cell), atol=1e-12)


def test_shape_quench_relaxes_anisotropic_and_shear_stress():
    from pamssw.cell_relax import CellRelaxer
    target_metric = np.diag([4.0, 5.0, 6.0])
    class MetricWell:
        def evaluate(self, state):
            cell = state.cell
            delta = cell @ cell.T - target_metric
            energy = 0.025 * np.sum(delta**2)
            stress = 0.1 * cell.T @ delta @ cell / np.linalg.det(cell)
            return EnergyResult(energy, np.zeros_like(state.positions), stress)
    state = bulk(np.array([[2.2, 0.3, 0.1], [0.0, 2.1, -0.2], [0.0, 0.0, 2.3]]))
    result = CellRelaxer(MetricWell(), mode='shape', stress_tol=1e-7).relax(
        state, fmax=1e-6, maxiter=150)
    assert result.telemetry.converged
    np.testing.assert_allclose(result.state.cell @ result.state.cell.T, target_metric, atol=3e-6)


def test_cell_relax_result_can_be_written_as_extxyz(tmp_path):
    from ase.io import read
    from pamssw.cell_relax import CellRelaxer
    from pamssw.io import write_state
    result = CellRelaxer(VolumeWell()).relax(bulk(), fmax=1e-4, maxiter=50)
    path = tmp_path / 'relaxed.extxyz'
    write_state(path, result.state)
    loaded = read(path)
    np.testing.assert_allclose(loaded.cell.array, result.state.cell)
    np.testing.assert_allclose(loaded.get_stress(voigt=False), result.state.metadata['stress'])
    assert loaded.info['enthalpy'] == pytest.approx(result.energy)
    assert loaded.info['potential_energy'] == pytest.approx(result.potential_energy)
    assert loaded.info['volume'] == pytest.approx(result.volume)
