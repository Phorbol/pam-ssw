from functools import partial
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from pamssw.standalone.constrained_reference import ConstrainedSSWConfig, run_constrained_ssw
from pamssw.standalone.ls_native_reference import NativeLSSettings
from pamssw.standalone.paper_reference import SSWConfig, run_ssw
from pamssw.standalone.periodic_ga import PeriodicCandidate
from pamssw.standalone.periodic_ga_reference import PeriodicGAConfig, run_periodic_ga
from pamssw.standalone.surface_ga import SurfaceCandidate, SurfaceProposal
from pamssw.standalone.surface_ga_reference import SurfaceGAConfig, run_surface_ga
from pamssw.standalone.surface import ASESurface


class Surface:
    requests = 0
    exhausted = False


def native_ls():
    return NativeLSSettings({(29, 29): 3.}, {(29, 29): 5.})


def periodic_config():
    return PeriodicGAConfig(quick_steps=1, generations=1, generation_steps=1,
        fine_steps=1, regions=1, fine_regions=1, min_ga=1, max_batches=1,
        max_cut_attempts=2, max_pair_attempts=2, partition_max_draws=10)


def periodic_walker_config():
    return SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
        temperature_K=300., fmax=.1, relax_steps=30, fd_step=.01,
        rotation_hvp=2, rotation_tol=.1, direction_sampling='global',
        cluster_frame='translation_only')


def periodic_kwargs(seeds):
    return dict(config=periodic_config(), walker_config=periodic_walker_config(),
        rng=np.random.default_rng(4), descriptor_basis=seeds,
        bond_lengths={(29, 29): 5.}, neighbor_range=1.1,
        projection_weights=[.3, .2, .2, .1, .1, .1], bond_limits={(29, 29): .5},
        matcher=lambda a, b: np.array_equal(a.positions, b.positions))


def assert_native_ls_records(result):
    assert result['walks'], 'GA returned no walks'
    for walk in result['walks']:
        records = list(getattr(walk['result'], 'records', ()))
        assert records, f"{walk['phase']} returned no SSW records"
        signal = False
        for step in records:
            if isinstance(step, dict):
                preparation = step.get('ls_preparation')
                update = step.get('native_ls_update', step.get('ls_update'))
                status = step.get('status')
            else:
                preparation, update, status = step.ls_preparation, step.ls_update, step.status
            signal |= (preparation is not None or update is not None or
                       status in {'ls_initialization_failed', 'ls_prequench_failed', 'ls_update_failed'})
        # A failed native prequench/update is explicit evidence for this walk;
        # an ordinary completed SSW record would mean the bound LS was dropped.
        assert signal, (walk['phase'], records)


def test_fixed_cell_periodic_ga_accepts_partial_ssw_with_native_ls_all_phases():
    seed = bulk('Cu', 'fcc', a=3.6, cubic=True)
    callback = partial(run_ssw, ls=native_ls())
    def proposal_factory(parents, energies, regions, rng, *, config, bond_limits):
        return SimpleNamespace(
            candidates=(PeriodicCandidate(parents[0].copy(), (0,), tuple(range(len(parents[0]))),
                                          'test', {}),), status='completed')
    result = run_periodic_ga([seed.copy(), seed.copy(), seed.copy()], ASESurface(EMT()),
        fixed_cell=True, walker=callback, proposal_factory=proposal_factory,
        **periodic_kwargs([seed] * 3))
    assert result['status'] in ('completed', 'censored')
    assert [walk['phase'] for walk in result['walks']] == [
        'quick', 'quick', 'quick', 'offspring_quick', 'fine']
    assert_native_ls_records(result)


def surface_config():
    return SurfaceGAConfig(quick_steps=1, generations=1, generation_steps=1,
        fine_steps=1, regions=2, fine_regions=1, min_ga=1, max_batches=1,
        max_cut_attempts=2, max_pair_attempts=2, max_face_attempts=2,
        max_insertion_attempts=2, partition_max_draws=10)


def constrained_walker_config():
    return ConstrainedSSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
                                fmax=.1, relax_steps=30)


def test_surface_ga_accepts_partial_constrained_ssw_with_native_ls_all_phases():
    seeds = [Atoms('Cu3', positions=[[0, 0, 0], [x, 0, 1], [x, 1, 1]],
                   cell=[10, 10, 10], pbc=[1, 1, 0]) for x in (1., 2., 3.)]
    callback = partial(run_constrained_ssw, ls=native_ls())
    def proposals(parents, energies, support, ads, rng, **kwargs):
        child = parents[0].copy()
        return SurfaceProposal((SurfaceCandidate(child, support, ads,
            tuple(range(len(child))), tuple(range(len(child))), 'test', {}),),
            'completed', (), 0)
    result = run_surface_ga(seeds, ASESurface(EMT()), config=surface_config(),
        walker_config=constrained_walker_config(), rng=np.random.default_rng(8),
        substrate_indices=[0], adsorbate_indices=[1, 2],
        routing=lambda a: [a.positions[1, 0], 0, 0],
        matcher=lambda a, b: np.array_equal(a.positions, b.positions),
        bond_limits={('Cu', 'Cu'): .5}, atomic_radii={29: 1.},
        site_fractional=(.5, .5), walker=callback, proposal_factory=proposals)
    assert result['status'] in ('completed', 'completed_with_failures')
    assert [walk['phase'] for walk in result['walks']] == [
        'quick', 'quick', 'quick', 'offspring_quick', 'fine']
    assert_native_ls_records(result)
