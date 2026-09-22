from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from pamssw.standalone import paper_ga
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
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


def _call_kwargs(surface, atoms=None, **extra):
    return dict(
        initial=[_atoms() if atoms is None else atoms], surface=surface,
        groups=((0, 1, 2), (3, 4, 5)), references=(0., 1., 2.),
        descriptor_bonds={(1, 1): 1.}, descriptor_weights=(1.,) * 6,
        neighbor_range=1.2, proposal_bond_limits={}, config=_config(),
        ssw_config=_ssw(), rng=np.random.default_rng(3), **extra)


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


def test_recovered_direction_checkpoint_same_settings_resume(monkeypatch):
    first, _ = _run(monkeypatch, direction=_direction(),
                    checkpoint_callback=lambda state: True)
    resumed, calls = _run(monkeypatch, direction=_direction(),
                          checkpoint=first.checkpoint)
    assert resumed.status in {'completed', 'completed_with_failures', 'no_eligible_minima'}
    assert calls


def test_default_checkpoint_contract_has_no_recovered_direction_key(monkeypatch):
    result, _ = _run(monkeypatch, checkpoint_callback=lambda state: True)
    assert result.checkpoint is not None
    assert 'recovered_direction' not in result.checkpoint.contract


def test_recovered_direction_rejects_rotation_conflict_before_surface():
    surface = SimpleNamespace(requests=0)
    rotation = RecoveredRotationSettings(5, 15, .2, .02, 'euclidean', 40)
    with pytest.raises(ValueError, match='mutually exclusive'):
        paper_ga.run_ga_ssw(**_call_kwargs(
            surface, recovered_direction=_direction(), recovered_rotation=rotation))
    assert surface.requests == 0


def test_recovered_direction_rejects_offspring_frame_and_prerotation_before_surface():
    surface = SimpleNamespace(requests=0)
    with pytest.raises(ValueError, match='direction_only'):
        paper_ga.run_ga_ssw(**_call_kwargs(
            surface, recovered_direction=_direction(),
            offspring_ssw_config=_ssw(cluster_frame='eckart')))
    assert surface.requests == 0
    with pytest.raises(ValueError, match='PreRot'):
        paper_ga.run_ga_ssw(**_call_kwargs(
            surface, recovered_direction=_direction(),
            ssw_config=_ssw(pre_rotation_hvp=1, rotation_bias=None, rotation_hvp=4)))
    assert surface.requests == 0


@pytest.mark.parametrize('mutate', [
    lambda atoms: atoms.set_constraint(FixAtoms(indices=[0])),
    lambda atoms: atoms.set_pbc(True),
])
def test_recovered_direction_rejects_constraints_and_pbc_before_surface(mutate):
    surface = SimpleNamespace(requests=0)
    atoms = _atoms()
    mutate(atoms)
    with pytest.raises(ValueError, match='free nonperiodic cluster'):
        paper_ga.run_ga_ssw(**_call_kwargs(
            surface, atoms=atoms, recovered_direction=_direction()))
    assert surface.requests == 0


def test_real_walker_constructs_and_initializes_a_fresh_controller_per_walk(monkeypatch):
    """The production driver, with only its expensive quench replaced, owns each controller."""
    from pamssw.standalone import paper_reference, recovered_direction

    created = []
    initialized = []
    base = recovered_direction.RecoveredDirectionController

    class SpyController(base):
        def __init__(self, settings):
            created.append(self)
            super().__init__(settings)

        def initialize(self, input_atoms, initial_quenched, rng):
            initialized.append(self)
            return super().initialize(input_atoms, initial_quenched, rng)

    class FlatSurface:
        requests = 0

        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    monkeypatch.setattr(recovered_direction, 'RecoveredDirectionController', SpyController)
    monkeypatch.setattr(paper_reference, 'quench',
                        lambda a, surface, **unused:
                        QuenchResult(a.copy(), 0., 0., True, 0, 0, 'true'))
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda *args: 0.)
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda *args: 0.)
    config = replace(_config(), quick_steps=0, generations=0,
                     generation_steps=0, fine_steps=0, offspring_steps=0)
    atoms = _atoms()
    result = paper_ga.run_ga_ssw(
        [atoms, atoms.copy()], surface=FlatSurface(),
        groups=((0, 1, 2), (3, 4, 5)), references=(0., 1., 2.),
        descriptor_bonds={(1, 1): 1.}, descriptor_weights=(1.,) * 6,
        neighbor_range=1.2, proposal_bond_limits={}, config=config,
        ssw_config=_ssw(), recovered_direction=_direction(),
        rng=np.random.default_rng(3))
    assert result.status in {'completed', 'completed_with_failures', 'no_eligible_minima'}
    assert len(created) == len(initialized) == 2
    assert created[0] is not created[1]
