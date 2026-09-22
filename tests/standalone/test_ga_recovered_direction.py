from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone import paper_ga
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult


def _atoms():
    return Atoms("OHHOHH", positions=np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [3., 0., 0.], [4., 0., 0.], [3., 1., 0.]]))


def _direction():
    return RecoveredDirectionSettings(
        ratio_local=50, local_probability=.5, group_threshold=.5,
        pre_rotmax=1, rotmax=1, pre_ftol=.01, ftol=.01,
        metric='euclidean', max_force_calls=8)


def _ssw(**changes):
    value = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
                      temperature_K=150., fmax=.1, relax_steps=1,
                      fd_step=.01, rotation_hvp=2, rotation_tol=.1,
                      direction_sampling='global', cluster_frame='direction_only')
    return replace(value, **changes)


def _config():
    return paper_ga.PaperGAConfig(
        quick_steps=1, generations=1, generation_steps=1, fine_steps=1,
        ga_candidates=4, regions=1, fine_regions=1, quench_fmax=.1,
        quench_steps=1, proposal_max_batches=1, proposal_max_cut_attempts=2,
        proposal_max_pair_attempts=2, partition_max_draws=10,
        projection_tolerance=1e-8, energy_window=100., offspring_steps=1)


def _run(monkeypatch, *, direction=None, checkpoint=None,
         checkpoint_callback=None, ssw_config=None):
    atoms = _atoms()
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'quench', lambda a, surface, **unused:
                        QuenchResult(a.copy(), 0., 0., True, 0, 0, 'true'))
    monkeypatch.setattr(paper_ga, 'partition', lambda *args, **kwargs: [[0]])
    monkeypatch.setattr(paper_ga, 'propose_type3', lambda *args, **kwargs: ProposalResult(
        (GeneticCandidate(atoms.copy(), ((0, 1, 2), (3, 4, 5)), 'fixture', (0, 0), {}),),
        'target_reached', 1, 0))
    calls = []

    def walker(a, surface, *, steps, **options):
        calls.append(options)
        q = QuenchResult(a.copy(), 0., 0., True, 0, 0, 'true')
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')

    monkeypatch.setattr(paper_ga, 'run_ssw', walker)
    result = paper_ga.run_ga_ssw(
        [atoms], surface=SimpleNamespace(requests=0),
        groups=((0, 1, 2), (3, 4, 5)), references=(0., 1., 2.),
        descriptor_bonds={(1, 1): 1.}, descriptor_weights=(1.,) * 6,
        neighbor_range=1.2, proposal_bond_limits={}, config=_config(),
        ssw_config=ssw_config or _ssw(), offspring_ssw_config=_ssw(),
        recovered_direction=direction, checkpoint=checkpoint,
        checkpoint_callback=checkpoint_callback, rng=np.random.default_rng(3))
    return result, calls


def test_recovered_direction_reaches_all_four_ga_walk_phases(monkeypatch):
    result, calls = _run(monkeypatch, direction=_direction())
    assert {stage.phase for stage in result.stages if stage.phase in {
        'quick', 'offspring_ssw', 'generation_short', 'fine'}} == {
        'quick', 'offspring_ssw', 'generation_short', 'fine'}
    assert len(calls) == 4
    assert all(call['recovered_direction'] == _direction() for call in calls)


def test_recovered_direction_default_preserves_old_walker_keywords(monkeypatch):
    _, calls = _run(monkeypatch)
    assert calls and all('recovered_direction' not in call for call in calls)


def test_recovered_direction_rejects_invalid_options_before_surface(monkeypatch):
    surface = SimpleNamespace(requests=0)
    atoms = _atoms()
    with pytest.raises(TypeError, match='RecoveredDirectionSettings'):
        paper_ga.run_ga_ssw(
            [atoms], surface=surface, groups=((0, 1, 2), (3, 4, 5)),
            references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
            descriptor_weights=(1.,) * 6, neighbor_range=1.2,
            proposal_bond_limits={}, config=_config(), ssw_config=_ssw(),
            recovered_direction=object(), rng=np.random.default_rng(3))
    assert surface.requests == 0

    with pytest.raises(ValueError, match='direction_only'):
        paper_ga.run_ga_ssw(
            [atoms], surface=surface, groups=((0, 1, 2), (3, 4, 5)),
            references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
            descriptor_weights=(1.,) * 6, neighbor_range=1.2,
            proposal_bond_limits={}, config=_config(),
            ssw_config=_ssw(cluster_frame='eckart'),
            recovered_direction=_direction(), rng=np.random.default_rng(3))
    assert surface.requests == 0


def test_recovered_direction_checkpoint_setting_mismatch_rejected(monkeypatch):
    first, _ = _run(monkeypatch, direction=_direction(),
                    checkpoint_callback=lambda state: True)
    changed = replace(_direction(), ratio_local=51)
    with pytest.raises(ValueError, match='scientific/input contract'):
        _run(monkeypatch, direction=changed, checkpoint=first.checkpoint)
