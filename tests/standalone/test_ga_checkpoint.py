from types import SimpleNamespace

import numpy as np
from ase import Atoms

from pamssw.standalone import paper_ga
from pamssw.standalone.ga_checkpoint import GACheckpoint
from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
from pamssw.standalone.surface import QuenchResult


def _config():
    return paper_ga.PaperGAConfig(
        quick_steps=1, generations=1, generation_steps=1, fine_steps=1,
        ga_candidates=4, regions=1, fine_regions=1, quench_fmax=.1,
        quench_steps=1, proposal_max_batches=1, proposal_max_cut_attempts=2,
        proposal_max_pair_attempts=2, partition_max_draws=10,
        projection_tolerance=1e-8, energy_window=100.)


def _atoms(x):
    return Atoms('OHHOHH', positions=np.array([
        [x, 0., 0.], [x + 1., 0., 0.], [x, 1., 0.],
        [x + 3., 0., 0.], [x + 4., 0., 0.], [x + 3., 1., 0.]]))


def _run(monkeypatch, surface, **kwargs):
    monkeypatch.setattr(paper_ga, 'cluster_descriptor',
                        lambda numbers, positions, *args: float(positions[0, 0]))
    monkeypatch.setattr(paper_ga, 'descriptor_similarity',
                        lambda descriptor, reference, weights: descriptor)
    def quench(atoms, surface, **unused):
        if hasattr(surface, 'evaluate'):
            surface.evaluate(atoms)
        else:
            target = getattr(surface, '_wrapped', surface)
            target.requests += 1
        x = float(atoms.positions[0, 0])
        return QuenchResult(atoms.copy(), 10. - x, 0., True, 1, 1, 'true')
    monkeypatch.setattr(paper_ga, 'quench', quench)
    def walk(atoms, surface, *, steps, **unused):
        if hasattr(surface, 'evaluate'):
            surface.evaluate(atoms)
        else:
            target = getattr(surface, '_wrapped', surface)
            target.requests += 1
        q = QuenchResult(atoms.copy(), 10. - float(atoms.positions[0, 0]), 0., True,
                         1, 1, 'true')
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    monkeypatch.setattr(paper_ga, 'partition',
                        lambda archive, regions, rng, max_draws: [list(range(len(archive)))])
    proposals = []
    def propose(parents, energies, groups, change_types, rng, **unused):
        proposals.append((tuple(energies), float(rng.random())))
        return ProposalResult((GeneticCandidate(_atoms(2.), groups, 'fixture', (0, 0), {}),),
                               'target_reached', 1, 0)
    monkeypatch.setattr(paper_ga, 'propose_type3', propose)
    defaults = dict(initial=[_atoms(0.)], surface=surface, groups=((0, 1, 2), (3, 4, 5)),
                    references=(0., 1., 2.), descriptor_bonds={(1, 1): 1.},
                    descriptor_weights=(1.,) * 6, neighbor_range=1.2,
                    proposal_bond_limits={}, config=_config(), ssw_config=SimpleNamespace(),
                    rng=np.random.default_rng(17))
    defaults.update(kwargs)
    return paper_ga.run_ga_ssw(**defaults), proposals


def test_boundary_resume_preserves_lineage_rng_and_cost(monkeypatch, tmp_path):
    full, full_proposals = _run(monkeypatch, SimpleNamespace(requests=0))
    saved = []
    partial_surface = SimpleNamespace(requests=0)
    partial, partial_proposals = _run(
        monkeypatch, partial_surface,
        checkpoint_callback=lambda state: saved.append(state) or state.phase == 'generation_complete')
    assert partial.checkpoint is not None
    assert partial.status == 'checkpoint_boundary'
    assert partial.checkpoint.phase == 'generation_complete'
    assert partial_proposals == full_proposals
    path = tmp_path / 'ga.chk'
    partial.checkpoint.save(path)
    loaded = GACheckpoint.load(path)
    resumed, resumed_proposals = _run(monkeypatch, SimpleNamespace(requests=0), checkpoint=loaded)
    assert resumed_proposals == []
    assert partial.evaluation_requests < full.evaluation_requests
    assert resumed.evaluation_requests == full.evaluation_requests
    assert [stage.phase for stage in resumed.stages] == [stage.phase for stage in full.stages]
    assert [obs.parent_ids for obs in resumed.observations] == [obs.parent_ids for obs in full.observations]
    assert resumed.status == full.status


def test_budget_exhaustion_is_terminal_and_not_a_checkpoint(monkeypatch):
    class PayingSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros((len(atoms), 3))
    result, _ = _run(monkeypatch, PayingSurface(), max_evaluations=1,
                     checkpoint_callback=lambda state: True)
    assert result.status == 'budget_exhausted'
    assert result.checkpoint is None


def test_zero_budget_rejects_no_requests(monkeypatch):
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda *args: 0.)
    def zero_quench(atoms, active, **kwargs):
        active.evaluate(atoms)
        raise AssertionError('zero budget must not evaluate')
    monkeypatch.setattr(paper_ga, 'quench', zero_quench)
    result = paper_ga.run_ga_ssw(
            [_atoms(0.)], surface, groups=((0, 1, 2), (3, 4, 5)), references=(0., 1., 2.),
            descriptor_bonds={(1, 1): 1.}, descriptor_weights=(1.,) * 6,
            neighbor_range=1.2, proposal_bond_limits={}, config=_config(),
            ssw_config=SimpleNamespace(), rng=np.random.default_rng(1), max_evaluations=0)
    assert result.status == 'budget_exhausted'
    assert result.evaluation_requests == 0
    assert result.checkpoint is None


def test_checkpoint_contract_mismatch_is_rejected_before_surface_request(monkeypatch):
    partial, _ = _run(monkeypatch, SimpleNamespace(requests=0),
                      checkpoint_callback=lambda state: True)
    surface = SimpleNamespace(requests=0)
    with np.testing.assert_raises(ValueError):
        _run(monkeypatch, surface, checkpoint=partial.checkpoint,
             neighbor_range=9.)
    assert surface.requests == 0


def test_resumed_budget_uses_cumulative_requests(monkeypatch):
    class PayingSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros((len(atoms), 3))
    first_surface = PayingSurface()
    partial, _ = _run(monkeypatch, first_surface, max_evaluations=3,
                      checkpoint_callback=lambda state: state.phase == 'quick_complete')
    assert partial.checkpoint is not None
    resumed_surface = PayingSurface()
    resumed, _ = _run(monkeypatch, resumed_surface, max_evaluations=3,
                      checkpoint=partial.checkpoint)
    assert resumed.evaluation_requests <= 3
    assert resumed.status == 'budget_exhausted'
    assert resumed.checkpoint is None
    assert resumed.evaluation_requests == 3 and resumed_surface.requests == 1


def test_cycle_boundary_resume_advances_to_next_cycle(monkeypatch):
    config = _config()
    config = paper_ga.PaperGAConfig(**{**config.__dict__, 'cycles': 2})
    partial, _ = _run(monkeypatch, SimpleNamespace(requests=0), config=config,
                      checkpoint_callback=lambda state: state.phase == 'cycle_complete')
    assert partial.status == 'checkpoint_boundary'
    assert partial.checkpoint.phase == 'cycle_complete'
    resumed, _ = _run(monkeypatch, SimpleNamespace(requests=0), config=config,
                      checkpoint=partial.checkpoint)
    assert resumed.status in ('completed', 'completed_with_failures')
    assert len(resumed.stages) > len(partial.stages)
