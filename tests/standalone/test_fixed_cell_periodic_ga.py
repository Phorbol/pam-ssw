import numpy as np
import pytest
from dataclasses import replace
from ase import Atoms
from ase.build import bulk

from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.periodic_ga_reference import PeriodicGAConfig, run_periodic_ga
from pamssw.standalone.molecular_periodic_ga_reference import run_molecular_periodic_ga
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.paper_reference import SSWResult
from pamssw.standalone.periodic_ga import PeriodicCandidate


def _ga_config():
    return PeriodicGAConfig(quick_steps=0, generations=0, generation_steps=0,
        fine_steps=0, regions=1, fine_regions=1, min_ga=1, max_batches=1,
        max_cut_attempts=10, max_pair_attempts=10, partition_max_draws=10)


def _ssw_config():
    return SSWConfig(width=.2, rotation_bias=.5, max_gaussians=1,
        temperature_K=300., fmax=.01, relax_steps=0, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global',
        cluster_frame='translation_only')


def _kwargs(seeds):
    return dict(config=_ga_config(), walker_config=_ssw_config(),
        rng=np.random.default_rng(4), descriptor_basis=seeds,
        bond_lengths={(29, 29): 2.6}, neighbor_range=1.1,
        projection_weights=[.3, .2, .2, .1, .1, .1],
        bond_limits={(29, 29): .5},
        matcher=lambda a, b: np.array_equal(a.positions, b.positions))


class _Surface:
    requests = 0
    exhausted = False


def _walker(atoms, surface, *, steps, config, rng):
    surface.requests += 1
    ev = QuenchResult(atoms=atoms.copy(), energy=-1., max_force=0.,
                      converged=True, optimizer_steps=0, evaluation_requests=1,
                      surface='true')
    return SSWResult(initial=ev, current=atoms.copy(), best=atoms.copy(),
                     minima=(ev,), records=(), evaluation_requests=1,
                     status='completed')


def test_fixed_cell_certificate_does_not_require_stress_fields():
    seed = bulk('Cu', 'fcc', a=3.6, cubic=True)
    result = run_periodic_ga([seed.copy(), seed.copy(), seed.copy()], _Surface(),
                             fixed_cell=True, walker=_walker, **_kwargs([seed] * 3))
    assert result['best']['certificate']['certificate_scope'] == 'fixed_cell_energy_maxforce'
    assert result['best']['objective'] == result['best']['energy']
    assert result['best']['certificate']['certified']


def test_fixed_cell_rejects_noncommon_parent_cell_before_walker():
    seeds = [bulk('Cu', 'fcc', a=a, cubic=True) for a in (3.6, 3.6, 3.7)]
    surface = _Surface()
    with pytest.raises(ValueError, match='common fixed cell|cell'):
        run_periodic_ga(seeds, surface, fixed_cell=True, walker=_walker,
                        **_kwargs(seeds))
    assert surface.requests == 0


def test_type2_fixed_cell_is_rejected_before_pes():
    seeds = [bulk('Cu', 'fcc', a=3.6, cubic=True) for _ in range(3)]
    surface = _Surface()
    with pytest.raises(ValueError, match='TYPE2|fixed.cell|fixed cell'):
        run_molecular_periodic_ga(seeds, surface, molecules=((0, 1), (2, 3)),
                                  fixed_cell=True, **_kwargs(seeds))
    assert surface.requests == 0


def test_fixed_cell_rejects_non_translation_sampling_before_walker():
    seed = bulk('Cu', 'fcc', a=3.6, cubic=True)
    surface = _Surface()
    bad = replace(_ssw_config(), cluster_frame='cartesian')
    args = _kwargs([seed] * 3)
    args['walker_config'] = bad
    with pytest.raises(ValueError, match='translation_only'):
        run_periodic_ga([seed.copy() for _ in range(3)], surface,
                        fixed_cell=True, walker=_walker, **args)
    assert surface.requests == 0


def test_fixed_cell_restores_type1_canonical_crossover_frame_before_walker():
    from ase.cell import Cell
    from types import SimpleNamespace
    seed = bulk('Cu', 'fcc', a=3.6, cubic=True)
    original = np.array([[0., 3.6, 0.], [0., 0., 3.6], [3.6, 0., 0.]])
    seeds = []
    for shift in (0., .01, .02):
        a = seed.copy(); a.set_cell(original, scale_atoms=False); a.positions += shift
        seeds.append(a)
    canonical = Cell.fromcellpar(Cell(original).cellpar()).array
    seen = []
    proposed = []
    def walker(atoms, surface, *, steps, config, rng):
        seen.append(atoms.copy())
        return _walker(atoms, surface, steps=steps, config=config, rng=rng)
    def factory(parents, energies, regions, rng, *, config, bond_limits):
        child = parents[0].copy(); child.set_cell(canonical, scale_atoms=False)
        proposed.append(child.copy())
        return SimpleNamespace(candidates=(PeriodicCandidate(child, (0,), tuple(range(len(child))), 'test', {}),),
                               status='completed')
    cfg = replace(_ga_config(), generations=1, regions=2, generation_steps=0)
    args = _kwargs(seeds); args['config'] = cfg
    result = run_periodic_ga(seeds, _Surface(), fixed_cell=True, walker=walker,
                             proposal_factory=factory, **args)
    assert result['walks'][3]['frame_transform'] == 'canonical_cellpar_to_common_cell'
    assert all(np.array_equal(a.cell.array, original) for a in seen)
    np.testing.assert_allclose(seen[3].get_all_distances(mic=True),
                               proposed[0].get_all_distances(mic=True), atol=1e-12, rtol=0)


def test_fixed_cell_bad_child_cell_is_rejected_before_child_walker():
    from types import SimpleNamespace
    seed = bulk('Cu', 'fcc', a=3.6, cubic=True)
    seeds = []
    for shift in (0., .01, .02):
        a = seed.copy(); a.positions += shift; seeds.append(a)
    seen = []
    def walker(atoms, surface, *, steps, config, rng):
        seen.append(atoms.copy())
        return _walker(atoms, surface, steps=steps, config=config, rng=rng)
    def factory(parents, energies, regions, rng, *, config, bond_limits):
        child = parents[0].copy(); child.cell[0, 0] *= 1.01
        return SimpleNamespace(candidates=(PeriodicCandidate(child, (0,), tuple(range(len(child))), 'test', {}),),
                               status='completed')
    cfg = replace(_ga_config(), generations=1, regions=2, generation_steps=0)
    args = _kwargs(seeds); args['config'] = cfg
    run_periodic_ga(seeds, _Surface(), fixed_cell=True,
                    walker=walker, proposal_factory=factory, **args)
    assert len(seen) == 4  # three quick walks plus fine; bad offspring never reaches walker
    assert all(np.array_equal(a.cell.array, seed.cell.array) for a in seen)


def test_fixed_cell_bad_true_surface_certificate_is_retained_unarchived():
    seed = bulk('Cu', 'fcc', a=3.6, cubic=True)
    def bad_walker(atoms, surface, *, steps, config, rng):
        result = _walker(atoms, surface, steps=steps, config=config, rng=rng)
        ev = replace(result.initial, surface='modified')
        return replace(result, initial=ev, minima=(ev,))
    result = run_periodic_ga([seed.copy() for _ in range(3)], _Surface(),
                             fixed_cell=True, walker=bad_walker,
                             **_kwargs([seed] * 3))
    assert result['observations'] and not result['archive']
    assert all(not item['certificate']['certified'] for item in result['observations'])
