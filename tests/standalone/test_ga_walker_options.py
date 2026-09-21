from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone import paper_ga
from pamssw.standalone.native_mc import NativeMCSettings
from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult


def _atoms():
    return Atoms("OHHOHH", positions=np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [3., 0., 0.], [4., 0., 0.], [3., 1., 0.]]))


def _ga_config():
    return paper_ga.PaperGAConfig(
        quick_steps=1, generations=0, generation_steps=0, fine_steps=0,
        ga_candidates=1, regions=1, fine_regions=1, quench_fmax=.1,
        quench_steps=1, proposal_max_batches=1, proposal_max_cut_attempts=2,
        proposal_max_pair_attempts=2, partition_max_draws=10,
        projection_tolerance=1e-8, energy_window=100.)


def _ssw_config(**changes):
    value = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
                      temperature_K=150., fmax=.1, relax_steps=1,
                      fd_step=.01, rotation_hvp=2, rotation_tol=.1,
                      direction_sampling='global', cluster_frame='direction_only')
    return replace(value, **changes)


def _invoke(monkeypatch, *, ssw_config=None, full=False, **kwargs):
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'quench', lambda atoms, surface, **unused:
                        QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'true'))
    calls = []

    def walker(atoms, surface, *, steps, **options):
        calls.append(dict(options, steps=steps))
        q = QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'true')
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')

    monkeypatch.setattr(paper_ga, 'run_ssw', walker)
    config = _ga_config()
    if full:
        config = replace(config, generations=1, generation_steps=1, fine_steps=1,
                         offspring_steps=1, ga_candidates=4)
        monkeypatch.setattr(paper_ga, 'partition',
                            lambda archive, regions, rng, max_draws: [[0]])
        monkeypatch.setattr(
            paper_ga, 'propose_type3',
            lambda *args, **options: ProposalResult(
                (GeneticCandidate(_atoms(), ((0, 1, 2), (3, 4, 5)),
                                  'fixture', (0, 0), {}),),
                'target_reached', 1, 0))
    result = paper_ga.run_ga_ssw(
        [_atoms()], surface=SimpleNamespace(requests=0), groups=((0, 1, 2), (3, 4, 5)),
        references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
        descriptor_weights=(1.,) * 6, neighbor_range=1.2, proposal_bond_limits={},
        config=config, ssw_config=ssw_config or _ssw_config(),
        rng=np.random.default_rng(3), **kwargs)
    return result, calls


def test_optional_mc_and_recovered_rotation_reach_every_walker(monkeypatch):
    mc = NativeMCSettings(.1, 99999)
    rotation = RecoveredRotationSettings(5, 15, .2, .02, 'euclidean', 40)
    _, calls = _invoke(monkeypatch, full=True, mc=mc, recovered_rotation=rotation,
                       offspring_ssw_config=_ssw_config())
    assert len(calls) == 4
    assert all(call['mc'] == mc for call in calls)
    assert all(call['recovered_rotation'] == rotation for call in calls)
    assert {call['steps'] for call in calls} == {1}


def test_default_walker_contract_omits_new_keywords(monkeypatch):
    _, calls = _invoke(monkeypatch)
    assert calls
    assert all('mc' not in call and 'recovered_rotation' not in call for call in calls)


def test_invalid_mc_is_rejected_before_surface_request(monkeypatch):
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda *args: 0.)
    with pytest.raises(TypeError, match='mc must be NativeMCSettings'):
        paper_ga.run_ga_ssw(
            [_atoms()], surface=surface, groups=((0, 1, 2), (3, 4, 5)),
            references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
            descriptor_weights=(1.,) * 6, neighbor_range=1.2, proposal_bond_limits={},
            config=_ga_config(), ssw_config=_ssw_config(), rng=np.random.default_rng(3),
            mc=object())
    assert surface.requests == 0


def test_mc_temperature_and_rotation_conflict_are_rejected_early(monkeypatch):
    mc = NativeMCSettings(.1, 99999)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda *args: 0.)
    with pytest.raises(ValueError, match='positive temperature'):
        paper_ga.run_ga_ssw(
            [_atoms()], surface=surface, groups=((0, 1, 2), (3, 4, 5)),
            references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
            descriptor_weights=(1.,) * 6, neighbor_range=1.2, proposal_bond_limits={},
            config=_ga_config(), ssw_config=_ssw_config(temperature_K=0.),
            rng=np.random.default_rng(3), mc=mc)
    assert surface.requests == 0

    with pytest.raises(ValueError, match='positive temperature'):
        paper_ga.run_ga_ssw(
            [_atoms()], surface=surface, groups=((0, 1, 2), (3, 4, 5)),
            references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
            descriptor_weights=(1.,) * 6, neighbor_range=1.2, proposal_bond_limits={},
            config=_ga_config(), ssw_config=_ssw_config(),
            offspring_ssw_config=_ssw_config(temperature_K=0.),
            rng=np.random.default_rng(3), mc=mc)
    assert surface.requests == 0

    rotation = RecoveredRotationSettings(5, 15, .2, .02, 'euclidean', 40)
    with pytest.raises(ValueError, match='pre_rotation_hvp'):
        paper_ga.run_ga_ssw(
            [_atoms()], surface=surface, groups=((0, 1, 2), (3, 4, 5)),
            references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
            descriptor_weights=(1.,) * 6, neighbor_range=1.2, proposal_bond_limits={},
            config=_ga_config(), ssw_config=_ssw_config(pre_rotation_hvp=1,
                                                        rotation_bias=None, rotation_hvp=40),
            rng=np.random.default_rng(3), recovered_rotation=rotation)
    assert surface.requests == 0


def test_checkpoint_rejects_changed_optional_walker_settings(monkeypatch):
    first, _ = _invoke(monkeypatch, mc=NativeMCSettings(.1, 99999),
                       checkpoint_callback=lambda state: True)
    assert first.checkpoint is not None
    with pytest.raises(ValueError, match='scientific/input contract'):
        _invoke(monkeypatch, mc=NativeMCSettings(.2, 99999),
                checkpoint=first.checkpoint)
