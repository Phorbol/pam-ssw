import numpy as np
from ase import Atoms
from ase.io import read
from pathlib import Path

from pamssw.standalone.periodic_direction import (
    generate_periodic_direction,
    select_periodic_direction_group,
)


def _atoms(positions, *, cell=((5, 0, 0), (1.2, 4.5, 0), (0, 0, 12)), pbc=(1, 1, 0)):
    return Atoms("C" * len(positions), positions=positions, cell=cell, pbc=pbc)


def test_group_selection_is_invariant_to_independent_lattice_images():
    positions = np.array([[.1, .1, 2], [1.4, .1, 2], [2.5, .8, 2],
                          [3.6, 1.1, 2], [4.2, 3.8, 2.]])
    atoms = _atoms(positions)
    shifted = atoms.copy()
    b = np.array([1.2, 4.5, 0])
    shifted.positions += np.array([[5, 0, 0], [-b[0], -b[1], 0],
                                   [-5 + b[0], b[1], 0], [2 * 5 - 2 * b[0], -2 * b[1], 0],
                                   [0, 0, 0]])
    a = select_periodic_direction_group(positions, atoms, iter([.3]))
    b = select_periodic_direction_group(shifted.positions, shifted, iter([.3]))
    assert a.pair == b.pair
    np.testing.assert_array_equal(a.group_mask, b.group_mask)


def test_symmetric_axis_tie_uses_lowest_index_without_extra_random_draw():
    positions = np.array([[0., 0., 2.], [3., 0., 2.], [7., 0., 2.]])
    atoms = _atoms(positions, cell=((10, 0, 0), (0, 10, 0), (0, 0, 12)))
    draws = iter([.4, .91])
    got = select_periodic_direction_group(positions, atoms, draws)
    # Atoms 1 and 2 are symmetry-equivalent axis maxima. Lowest index wins;
    # the only RNG draw selects atom 2 from the score band.
    assert got.pair == (1, 2)
    assert next(draws) == .91


def test_stage_reference_uses_continuous_displacement_and_fixed_second_endpoint():
    positions = np.column_stack([np.arange(6, dtype=float), np.zeros(6), np.full(6, 2.)])
    atoms = _atoms(positions, cell=((10, 0, 0), (0, 10, 0), (0, 0, 12)))
    reference = positions.copy()
    # Atom 0 has the smallest raw movement, but it is fixed. Atom 5 is the
    # sole active first-axis candidate. The score-band second endpoint may be
    # fixed and is selected from the full geometry.
    reference[0, 0] -= 6.
    active = np.array([False, False, False, False, False, True])
    got = select_periodic_direction_group(reference, atoms, lambda: 0., active)
    assert got.pair[0] == 5
    assert got.pair[1] is not None and not active[got.pair[1]]
    assert np.all(got.group_mask[~active] == 0)


def test_stage_displacement_over_half_cell_is_not_minimum_imaged():
    positions = np.array([[.1, .1, 2], [2., .1, 2], [4., .1, 2],
                          [6., .1, 2], [8., .1, 2]])
    atoms = _atoms(positions)
    reference = positions.copy()
    reference[0, 0] += 4.8
    # The continuous displacement is 4.8 A even though its MIC is 0.2 A.
    # Keep another atom at 0.3 A movement so a MIC-based rank would reverse.
    reference[1, 0] -= .3
    active = np.array([True, True, False, False, False])
    from pamssw.standalone.periodic_direction import _stage_movement
    movement = _stage_movement(positions, reference)
    np.testing.assert_allclose(movement[:2], [4.8, .3], atol=1e-14)
    got = select_periodic_direction_group(reference, atoms, lambda: 0., active)
    assert got.pair[0] in (0, 1)
    # The geometry result remains finite while retaining the continuous chart.
    assert np.isfinite(got.group_mask).all()


def test_c6_uses_one_rooted_periodic_image_chart_and_is_representation_invariant():
    positions = np.array([[.1, .2, 2], [1.4, .3, 2], [4.8, .4, 2],
                          [2.0, 1.8, 2]])
    atoms = _atoms(positions)
    shifted = atoms.copy()
    b = np.array([1.2, 4.5, 0])
    shifted.positions += np.array([[0, 0, 0], [5, 0, 0], [-5, 0, 0],
                                   [-b[0], -b[1], 0]])
    coeff = np.zeros(10); coeff[6] = 1.
    seed = np.zeros_like(positions)
    one = generate_periodic_direction(atoms, seed, coeff, (0, 1), [0, 0, 1, 1],
                                      iter([.4]), group_marker=0)
    two = generate_periodic_direction(shifted, seed, coeff, (0, 1), [0, 0, 1, 1],
                                      iter([.4]), group_marker=0)
    np.testing.assert_allclose(one.direction, two.direction, atol=1e-12)
    assert one.local_route == "torsion"


def test_periodic_c1_radius_follows_images_and_active_mask_stays_supported():
    atoms = _atoms([[.1, .1, 2], [4.9, .1, 2], [2.2, 1.4, 2], [3.5, 3, 2]])
    active = np.array([True, True, False, True])
    coeff = np.zeros(10); coeff[1] = 1.
    got = generate_periodic_direction(atoms, np.zeros((4, 3)), coeff, (0, 1),
                                      np.zeros(4, dtype=int), iter([.2]),
                                      group_marker=0, active_mask=active)
    np.testing.assert_array_equal(got.direction[~active], 0.)
    assert np.linalg.norm(got.direction) == 1.


def test_displacement_seed_is_not_minimum_imaged_and_frozen_support_is_zeroed():
    atoms = _atoms([[.1, .1, 2], [1.3, .4, 2], [2.1, 1.1, 2]])
    seed = np.array([[2.0, 0, 0], [-1., 0, 0], [3., 0, 0]])
    coeff = np.zeros(10); coeff[9] = 1.
    active = np.array([True, True, False])
    got = generate_periodic_direction(atoms, seed, coeff, (0, 1), np.zeros(3, int),
                                      iter([.8]), group_marker=0, active_mask=active)
    expected = np.array([[2., 0, 0], [-1., 0, 0], [0., 0, 0]])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(got.direction, expected, atol=1e-12)


def test_skew_cell_two_dimensional_periodicity_and_connected_c4_fallback():
    # The endpoints are nearest images across the skew-cell a-vector and are
    # connected under the recovered C-C cutoff, so c4 takes pair fallback.
    atoms = _atoms([[.05, .05, 2], [4.85, 4.4, 2], [2.2, 2.2, 2]],
                   cell=((5, 0, 0), (1.2, 4.5, 0), (0, 0, 12)), pbc=(1, 1, 0))
    coeff = np.zeros(10); coeff[4] = 1.
    got = generate_periodic_direction(atoms, np.zeros((3, 3)), coeff, (0, 1),
                                      np.zeros(3, int), lambda: .7, group_marker=-1)
    assert got.local_route == "pair_fallback"
    assert got.group_marker == 0
    assert np.all(np.isfinite(got.direction))


def test_tio2_rutile_structure_exercises_periodic_local_geometry():
    source = (Path(__file__).parents[2] / "research/ga_ssw/evidence"
              / "tio2-native-ls-paired-20260924/inputs/phase87.extxyz")
    atoms = read(source)
    selected = select_periodic_direction_group(atoms.positions.copy(), atoms, lambda: .25)
    assert selected.pair[1] is not None
    assert selected.group_mask.shape == (len(atoms),)
    coeff = np.zeros(10); coeff[6] = 1.
    group = np.ones(len(atoms), dtype=np.int32)
    got = generate_periodic_direction(atoms, np.zeros((len(atoms), 3)), coeff,
                                      (0, 1), group, lambda: .25, group_marker=0)
    assert got.local_route == "torsion"
    assert not got.release_all
    assert np.isfinite(got.direction).all()
