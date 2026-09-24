from types import SimpleNamespace
from copy import deepcopy

import numpy as np
from ase import Atoms

from pamssw.standalone import paper_ga
from pamssw.standalone.ga_checkpoint import GACheckpoint
from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.paper_reference import SSWCheckpoint, SSWStep


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
    step_walk = kwargs.pop('step_walk', False)
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
        if step_walk:
            def pay():
                if hasattr(surface, 'evaluate'):
                    surface.evaluate(atoms)
                else:
                    surface.requests += 1
            saved = unused.get('checkpoint')
            rng = unused['rng']
            if saved is None:
                pay()
                q = QuenchResult(atoms.copy(), 10. - float(atoms.positions[0, 0]),
                                 0., True, 1, 1, 'true')
                records = []
                start, paid = 0, 1
            else:
                rng.bit_generator.state = deepcopy(saved.rng_state)
                q = saved.initial
                records = list(saved.records)
                start, paid = saved.next_index, saved.evaluation_requests
            for index in range(start, start + steps):
                pay()
                records.append(SSWStep(index, 'accepted', True, (float(rng.random()),),
                                       None, None, 1, last_atoms=atoms.copy()))
                paid += 1
                cp = SSWCheckpoint(q, atoms.copy(), q.energy, q, (q,), tuple(records),
                    None, None, unused['config'], unused.get('ls'),
                    unused.get('height_policy'), unused.get('gaussian_policy'),
                    unused.get('height_update_budget', 1000), None,
                    deepcopy(rng.bit_generator.state), paid, index + 1, 'completed')
                callback = unused.get('checkpoint_callback')
                if callback is not None and callback(cp):
                    return SimpleNamespace(initial=q, minima=(q,), records=tuple(records),
                        evaluation_requests=paid, status='paused', checkpoint=cp)
            return SimpleNamespace(initial=q, minima=(q,), records=tuple(records),
                evaluation_requests=paid, status='completed')
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


def test_active_walk_resume_all_phases_preserves_queue_rng_and_cost(monkeypatch, tmp_path):
    config = paper_ga.PaperGAConfig(**{**_config().__dict__,
        'quick_steps': 2, 'generation_steps': 2, 'fine_steps': 2,
        'offspring_steps': 2})
    full, full_proposals = _run(monkeypatch, SimpleNamespace(requests=0),
        config=config, step_walk=True, checkpoint_walk_steps=True)
    for phase in ('quick', 'offspring_ssw', 'generation_short', 'fine'):
        first_surface = SimpleNamespace(requests=0)
        first, first_proposals = _run(monkeypatch, first_surface,
            config=config, step_walk=True, checkpoint_walk_steps=True,
            checkpoint_callback=lambda cp: cp.phase == 'active_walk' and cp.active_walk.phase == phase)
        assert first.status == 'checkpoint_boundary'
        assert first.checkpoint.active_walk.phase == phase
        assert first.checkpoint.active_walk.ssw_checkpoint.next_index == 1
        path = tmp_path / f'{phase}.pkl'
        first.checkpoint.save(path)
        second_surface = SimpleNamespace(requests=0)
        resumed, second_proposals = _run(monkeypatch, second_surface,
            config=config, step_walk=True, checkpoint_walk_steps=True,
            checkpoint=GACheckpoint.load(path))
        assert first_surface.requests + second_surface.requests == full.evaluation_requests
        assert resumed.evaluation_requests == full.evaluation_requests
        assert first_proposals + second_proposals == full_proposals
        assert [(s.phase, s.seed_id, s.evaluation_requests) for s in resumed.stages] == [
            (s.phase, s.seed_id, s.evaluation_requests) for s in full.stages]
        assert [(o.phase, o.id, o.parent_ids, o.operator) for o in resumed.observations] == [
            (o.phase, o.id, o.parent_ids, o.operator) for o in full.observations]
        assert [r.climb for w in resumed.walks for r in w.records] == [
            r.climb for w in full.walks for r in w.records]


def test_active_walk_rejects_incompatible_nested_state_before_pes(monkeypatch):
    config = paper_ga.PaperGAConfig(**{**_config().__dict__, 'quick_steps': 2})
    paused, _ = _run(monkeypatch, SimpleNamespace(requests=0), config=config,
        step_walk=True, checkpoint_walk_steps=True,
        checkpoint_callback=lambda cp: cp.phase == 'active_walk')
    corrupt = paused.checkpoint.clone()
    corrupt.active_walk.ssw_checkpoint.config = SimpleNamespace(bad=True)
    surface = SimpleNamespace(requests=0)
    with np.testing.assert_raises(ValueError):
        _run(monkeypatch, surface, config=config, step_walk=True,
             checkpoint_walk_steps=True, checkpoint=corrupt)
    assert surface.requests == 0


def test_v1_completed_boundary_still_resumes(monkeypatch, tmp_path):
    paused, _ = _run(monkeypatch, SimpleNamespace(requests=0),
        checkpoint_callback=lambda cp: cp.phase == 'quick_complete')
    old = paused.checkpoint.clone()
    old.version = 1
    del old.active_walk
    path = tmp_path / 'v1.pkl'
    old.save(path)
    loaded = GACheckpoint.load(path)
    resumed, _ = _run(monkeypatch, SimpleNamespace(requests=0), checkpoint=loaded)
    assert resumed.status in ('completed', 'completed_with_failures')


def test_real_emt_ls_active_quick_walk_resume(monkeypatch):
    from ase.calculators.emt import EMT
    from pamssw.standalone.surface import ASESurface
    from test_ssw_checkpoint import _case

    atoms, ssw_config, ls = _case()
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda *args: 0.)
    config = paper_ga.PaperGAConfig(**{**_config().__dict__,
        'proposal_type': 0, 'quick_steps': 2, 'generations': 0,
        'generation_steps': 0, 'fine_steps': 0, 'quench_fmax': 1e-5,
        'quench_steps': 100})
    common = dict(initial=[atoms], groups=None, references=(0., 1., 2.),
        descriptor_bonds={}, descriptor_weights=(1.,) * 6, neighbor_range=1.,
        proposal_bond_limits={}, config=config, ssw_config=ssw_config, ls=ls)
    full = paper_ga.run_ga_ssw(surface=ASESurface(EMT()), rng=np.random.default_rng(19),
        checkpoint_walk_steps=True, checkpoint_callback=lambda cp: False, **common)
    first_surface = ASESurface(EMT())
    first = paper_ga.run_ga_ssw(surface=first_surface, rng=np.random.default_rng(19),
        checkpoint_walk_steps=True,
        checkpoint_callback=lambda cp: cp.phase == 'active_walk', **common)
    assert first.checkpoint.active_walk.ssw_checkpoint.response.steps == 1
    second_surface = ASESurface(EMT())
    resumed_rng = np.random.default_rng(999)
    resumed = paper_ga.run_ga_ssw(surface=second_surface, rng=resumed_rng,
        checkpoint_walk_steps=True, checkpoint=first.checkpoint, **common)
    assert resumed.evaluation_requests == full.evaluation_requests
    assert first_surface.requests + second_surface.requests == full.evaluation_requests
    assert resumed.walks[0].checkpoint.response.steps == full.walks[0].checkpoint.response.steps
    assert [r.ls_update for r in resumed.walks[0].records] == [r.ls_update for r in full.walks[0].records]
    np.testing.assert_array_equal(resumed.walks[0].current.positions, full.walks[0].current.positions)
