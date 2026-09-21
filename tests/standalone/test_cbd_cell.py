"""Real Cu/EMT cell geometry checks; not evidence of search efficiency."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.cbd_cell import CellChart, cell_direction


def reference():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.set_cell([[3.6, .18, -.07], [.06, 3.7, .13], [-.11, .04, 3.55]], scale_atoms=True)
    atoms.positions += np.random.default_rng(12).normal(scale=.027, size=(4, 3))
    atoms.positions[0] += 2 * atoms.cell[0]  # Preserve lifted image, no wrapping.
    return atoms


@pytest.mark.parametrize('pressure', [0., .035])
def test_full_nine_component_cell_gradient_at_finite_deformation(pressure):
    atoms = reference(); chart = CellChart(atoms)
    q = (atoms.cell.array @ np.array([[1.07, .04, -.02], [-.03, .94, .02], [.01, -.04, 1.03]])).ravel()
    surface = ASEStressSurface(EMT())
    result = chart.evaluate(q, surface.evaluate, pressure=pressure)
    for i in range(9):
        delta = np.zeros(9); delta[i] = 1e-5
        ep = chart.evaluate(q + delta, surface.evaluate, pressure=pressure).objective
        em = chart.evaluate(q - delta, surface.evaluate, pressure=pressure).objective
        assert (ep - em) / 2e-5 == pytest.approx(result.gradient[i], abs=2e-7)
    np.testing.assert_allclose(chart.pack(result.atoms), q, atol=2e-14)
    np.testing.assert_allclose(result.atoms.get_scaled_positions(wrap=False), atoms.get_scaled_positions(wrap=False), atol=2e-14)
    physical = surface.evaluate(result.atoms)
    np.testing.assert_allclose(result.forces, physical[1], atol=1e-13)
    np.testing.assert_allclose(result.stress, physical[2], atol=1e-13)


def test_rotation_projector_is_fixed_center_orthogonal_and_rank_six():
    atoms = reference(); chart = CellChart(atoms); q = chart.pack(atoms)
    q = q + np.arange(9) * .017
    p = np.column_stack([chart.project(np.eye(9)[i], center=q) for i in range(9)])
    np.testing.assert_allclose(p, p.T, atol=2e-15)
    np.testing.assert_allclose(p @ p, p, atol=2e-15)
    assert np.linalg.matrix_rank(p, tol=1e-12) == 6
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a = np.zeros((3, 3)); a[i, j] = 1; a[j, i] = -1
        np.testing.assert_allclose(chart.project((q.reshape(3, 3) @ a).ravel(), center=q), 0., atol=3e-15)
    result = chart.evaluate(q, ASEStressSurface(EMT()).evaluate, pressure=.035)
    np.testing.assert_allclose(p @ result.gradient, result.gradient, atol=2e-13)


def test_direction_real_cell_cost_rotation_and_positive_curvature_convention():
    atoms = reference(); chart = CellChart(atoms); q = chart.pack(atoms)
    surface = ASEStressSurface(EMT())
    anchor = np.random.default_rng(9).normal(size=9)
    mode = cell_direction(chart, q, anchor, evaluate=surface.evaluate, rotation_force_tol=1e-10)
    assert surface.requests == mode.force_calls <= 6
    assert mode.hvp_calls <= 5
    assert not mode.converged  # This deliberately tight residual budget is exhausted.
    assert np.linalg.norm(mode.direction) == pytest.approx(1.)
    np.testing.assert_allclose(chart.project(mode.direction, center=q), mode.direction, atol=2e-14)
    g0 = chart.evaluate(q, surface.evaluate).gradient
    g1 = chart.evaluate(q + .005 * mode.direction, surface.evaluate).gradient
    expected = mode.direction @ chart.project((g1 - g0) / .005, center=q)
    assert mode.curvature == pytest.approx(expected, abs=1e-12)
    assert mode.curvature > 0


def test_paper_force_tolerance_maps_to_gradient_hvp_residual():
    atoms = reference(); chart = CellChart(atoms); surface = ASEStressSurface(EMT())
    mode = cell_direction(chart, chart.pack(atoms), np.ones(9), evaluate=surface.evaluate)
    assert mode.converged == (mode.residual_norm <= .1 / (2 * .005))
    assert surface.requests <= 6


def test_invalid_chart_and_direction_fail_before_oracle():
    atoms = reference(); chart = CellChart(atoms); q = chart.pack(atoms)
    changed = atoms.copy(); changed.positions[0, 0] += .01
    with pytest.raises(ValueError, match='fractional'): chart.pack(changed)
    with pytest.raises(ValueError): chart.unpack(np.zeros(9))
    a = np.zeros((3, 3)); a[0, 1] = 1; a[1, 0] = -1
    surface = ASEStressSurface(EMT())
    with pytest.raises(ValueError, match='nonrotational'):
        cell_direction(chart, q, (atoms.cell.array @ a).ravel(), evaluate=surface.evaluate)
    assert surface.requests == 0
    atoms.pbc = False
    with pytest.raises(ValueError): CellChart(atoms)
