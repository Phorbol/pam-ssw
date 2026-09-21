"""Independent structure identity is separate from descriptor routing."""
from types import SimpleNamespace

import numpy as np
import pytest

from pamssw.standalone import paper_ga
from pamssw.standalone.surface import QuenchResult
from test_paper_ga import configuration, geometry


def run(initial, surface, matcher, monkeypatch, descriptor_fn=None):
    monkeypatch.setattr(paper_ga, 'cluster_descriptor',
                        descriptor_fn or (lambda numbers, positions, *args: float(positions[0, 0])))
    monkeypatch.setattr(paper_ga, 'descriptor_similarity',
                        lambda descriptor, reference, weights: descriptor)
    def fake_quench(atoms, surface, **kwargs):
        surface.requests += 1
        return QuenchResult(atoms.copy(), float(atoms.positions[0, 0]), 0., True, 0, 1, 'true')
    monkeypatch.setattr(paper_ga, 'quench', fake_quench)
    config = configuration(quick_steps=0, generations=0, fine_steps=0)
    return paper_ga.run_ga_ssw(
        initial, surface, groups=((0, 1, 2), (3, 4, 5)), references=(0., 1., 2.),
        descriptor_bonds={}, descriptor_weights=(1.,) * 6, neighbor_range=1.2,
        proposal_bond_limits={}, config=config, ssw_config=object(),
        rng=np.random.default_rng(3), structure_matcher=matcher)


def pair_distances(atoms):
    positions = atoms.positions
    return np.sort(np.linalg.norm(positions[:, None] - positions[None, :], axis=2).ravel())


def test_projection_collision_does_not_merge_when_matcher_rejects(monkeypatch):
    surface = SimpleNamespace(requests=0)
    result = run([geometry(0.), geometry(1.)], surface, lambda a, b: False, monkeypatch,
                 descriptor_fn=lambda numbers, positions, *args: 0.)
    assert result.identity_mode == 'caller_structure_matcher'
    assert len(result.archive) == 2


def test_default_projection_identity_merges_the_same_collision(monkeypatch):
    surface = SimpleNamespace(requests=0)
    result = run([geometry(0.), geometry(1.)], surface, None, monkeypatch,
                 descriptor_fn=lambda numbers, positions, *args: 0.)
    assert result.identity_mode == 'projection'
    assert len(result.archive) == 1


def test_matcher_merges_rigid_translation_and_keeps_lower_energy(monkeypatch):
    surface = SimpleNamespace(requests=0)
    result = run([geometry(1.), geometry(0.)], surface,
                 lambda a, b: np.allclose(pair_distances(a), pair_distances(b)), monkeypatch)
    assert len(result.archive) == 1
    assert result.archive[0]['energy'] == 0.


def test_matcher_exception_rejects_incoming_observation(monkeypatch):
    surface = SimpleNamespace(requests=0)
    calls = []
    def broken(a, b):
        calls.append(1)
        raise RuntimeError('identity backend failed')
    result = run([geometry(0.), geometry(1.)], surface, broken, monkeypatch)
    assert len(result.archive) == 1
    rejected = [o for o in result.observations if not o.eligible_for_archive]
    assert len(rejected) == 1
    assert 'identity matcher' in result.failures[0].reason
    assert result.evaluation_requests == surface.requests == 2
    assert len(calls) == 1
