import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.protected_move import ScalarMovePolicy, protected_move


def pair(distance=1.0):
    return Atoms('H2', positions=[[0, 0, 0], [distance, 0, 0]])


def test_direct_move_preserves_requested_width_and_direction():
    atoms = pair()
    direction = np.array([[0., 0., 0.], [0., 1., 0.]])
    result = protected_move(atoms, direction, .2, ScalarMovePolicy())
    assert result.accepted
    assert result.diagnostics['retry_count'] == 0
    assert result.width == pytest.approx(.2)
    np.testing.assert_allclose(result.direction, direction)
    np.testing.assert_allclose(result.atoms.positions - atoms.positions, .2 * direction)


def test_rejected_contraction_shrinks_and_reports_actual_width():
    atoms = pair()
    direction = np.array([[0., 0., 0.], [-1., 0., 0.]])
    result = protected_move(atoms, direction, .4, ScalarMovePolicy())
    assert result.accepted
    assert result.diagnostics['retry_count'] > 0
    assert result.width < .4
    assert result.width == pytest.approx(np.linalg.norm(result.atoms.positions - atoms.positions))
    assert result.diagnostics['attempts'][0]['minimum_separation_after'] == pytest.approx(.6)


def test_exhaustion_returns_saved_center_not_last_rejected_trial():
    atoms = pair()
    direction = np.array([[0., 0., 0.], [0., 1., 0.]])
    result = protected_move(atoms, direction, 1.,
        ScalarMovePolicy(displacement_max=.1, max_trials=2))
    assert not result.accepted
    assert result.diagnostics['reason'] == 'move_retry_exhausted'
    assert len(result.diagnostics['attempts']) == 2
    np.testing.assert_array_equal(result.atoms.positions, atoms.positions)


def test_periodic_self_image_sets_nonzero_minimum_separation():
    atoms = Atoms('H', positions=[[.1, .1, .1]], cell=[2., 2., 2.], pbc=True)
    direction = np.array([[0., 0., 1.]])
    result = protected_move(atoms, direction, .2, ScalarMovePolicy())
    assert result.accepted
    assert result.diagnostics['attempts'][0]['minimum_separation_before'] == pytest.approx(2.)


def test_zero_initial_separation_is_rejected_instead_of_guarded_as_safe():
    atoms = pair(0.)
    with pytest.raises(ValueError, match='positive initial minimum separation'):
        protected_move(atoms, np.ones((2, 3)), .1, ScalarMovePolicy())


def test_initial_width_below_native_retry_floor_is_still_evaluated():
    atoms = pair()
    direction = np.array([[0., 0., 0.], [0., 1., 0.]])
    result = protected_move(atoms, direction, .05, ScalarMovePolicy())
    assert result.accepted
    assert result.diagnostics['retry_count'] == 0
    assert result.width == pytest.approx(.05)


def test_native_retry_counter_boundary_is_fifty_total_trials():
    atoms = pair()
    direction = np.array([[0., 0., 0.], [0., 1., 0.]])
    result = protected_move(atoms, direction, 1., ScalarMovePolicy(
        displacement_max=.01, shrink_factor=.999, max_trials=50))
    assert not result.accepted
    assert len(result.diagnostics['attempts']) == 50


def test_move_policy_is_exported_and_checkpoint_schema9_preserves_wrapped_q_identity(tmp_path, monkeypatch):
    import pickle
    from ase.cluster import Icosahedron
    from types import SimpleNamespace
    import pamssw.standalone.paper_reference as reference
    import pamssw.standalone.recovered_cbd as cbd
    from pamssw.standalone import ScalarMovePolicy as PublicPolicy
    from pamssw.standalone.paper_reference import SSWConfig, load_ssw_checkpoint, run_ssw
    from pamssw.standalone.descriptor_direction import DescriptorMode
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.surface import QuenchResult

    assert PublicPolicy is ScalarMovePolicy
    atoms = Icosahedron('Cu', 2)
    atoms.positions[1] += [.17, -.08, .05]
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e9, relax_steps=2, fd_step=1e-4,
        rotation_hvp=4, rotation_tol=1e-3, direction_sampling='global',
        cluster_frame='direction_only')
    direction = RecoveredDirectionSettings(ratio_local=5, local_probability=1.,
        group_threshold=.5, pre_rotmax=2, rotmax=4, pre_ftol=.01, ftol=.01,
        metric='euclidean', max_force_calls=8,
        descriptor_modes=(DescriptorMode('S2', 0, (29,), 5., 2, 0., 1),),
        descriptor_probability=1.)
    path = tmp_path / 'protected-q.pkl'
    class FlatSurface:
        requests = 0
        def evaluate(self, value):
            self.requests += 1
            return 0., np.zeros_like(value.positions)
    def passthrough(value, surface, *, fmax, steps, terms=(), **kwargs):
        return QuenchResult(value.copy(), 0., 0., True, 0, 0, 'flat')
    monkeypatch.setattr(reference, 'quench', passthrough)
    def identity_cbd(atoms, anchor, **kwargs):
        return SimpleNamespace(direction=np.asarray(anchor).copy(), bias_reference=np.asarray(anchor).copy(),
            rotation_weight=0., stage='CBD_unbiasedRot', stage_complete=True,
            real_curvature=0., trace=(), curvature=0., residual_norm=0.,
            force_calls=0, stop_reason='converged', converged=True)
    monkeypatch.setattr(cbd, 'recovered_cbd_direction', identity_cbd)
    surface = FlatSurface()
    continuous = run_ssw(atoms, surface, steps=2, config=config,
        rng=np.random.default_rng(31), recovered_direction=direction,
        move_policy=ScalarMovePolicy(), checkpoint_path=tmp_path / 'continuous.pkl')
    split = run_ssw(atoms, FlatSurface(), steps=1, config=config,
        rng=np.random.default_rng(31), recovered_direction=direction,
        move_policy=ScalarMovePolicy(), checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert split.checkpoint.next_index == 1
    assert checkpoint.schema_version == 9
    assert checkpoint.wrapped_schema_version == 8
    assert checkpoint.base_schema_version == 4
    assert checkpoint.protected_move_policy == ScalarMovePolicy()
    resumed = run_ssw(atoms, FlatSurface(), steps=1, config=config,
        rng=np.random.default_rng(1), move_policy=ScalarMovePolicy(), checkpoint=checkpoint)
    assert resumed.status == continuous.status == 'completed'
    protected_stages = [event for record in continuous.records for event in record.climb
                        if 'protected_move' in event]
    assert protected_stages
    assert all(event['protected_move']['actual_width'] > 0 for event in protected_stages)
    assert pickle.dumps(resumed.checkpoint.rng_state) == pickle.dumps(continuous.checkpoint.rng_state)
    assert [x.status for x in resumed.records] == [x.status for x in continuous.records]
    assert resumed.evaluation_requests == continuous.evaluation_requests
    np.testing.assert_allclose(resumed.current.positions, continuous.current.positions)
    surface = FlatSurface()
    with pytest.raises(ValueError, match='protected move policy'):
        run_ssw(atoms, surface, steps=0, config=config,
            rng=np.random.default_rng(3), recovered_direction=direction,
            move_policy=ScalarMovePolicy(shrink_factor=.9), checkpoint=checkpoint)
    assert surface.requests == 0


def test_exhausted_driver_restores_stage_center_and_runs_true_landing_quench(monkeypatch):
    from types import SimpleNamespace
    from ase.calculators.emt import EMT
    import pamssw.standalone.dimer as dimer
    import pamssw.standalone.paper_reference as reference
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import ASESurface
    original_quench = reference.quench
    true_terms = []
    def capture_quench(*args, terms=(), **kwargs):
        true_terms.append(tuple(terms))
        return original_quench(*args, terms=terms, **kwargs)
    monkeypatch.setattr(reference, 'quench', capture_quench)

    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    config = SSWConfig(width=.2, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e9, relax_steps=2, fd_step=1e-4,
        rotation_hvp=4, rotation_tol=1e-3, direction_sampling='global',
        rotation_solver='dimer')
    direction = np.array([[0., 0., 0.], [0., 1., 0.]])
    mode = SimpleNamespace(direction=direction, converged=True, curvature=0.,
        residual_norm=0., force_calls=0, stop_reason='converged')
    monkeypatch.setattr(dimer, 'paper_dimer_direction', lambda *args, **kwargs: mode)
    result = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
        rng=np.random.default_rng(8),
        move_policy=ScalarMovePolicy(displacement_max=.01, max_trials=1))
    record = result.records[0]
    assert record.status == 'move_retry_exhausted'
    assert record.landing is not None and record.landing.converged
    assert true_terms[-1] == ()
    np.testing.assert_allclose(record.landing.atoms.positions, result.initial.atoms.positions,
        atol=1e-10)
    assert record.climb[-1]['move_diagnostics']['reason'] == 'move_retry_exhausted'


def test_accepted_shortened_width_and_direction_are_used_by_persistent_gaussian(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.dimer as dimer
    import pamssw.standalone.paper_reference as reference
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import QuenchResult

    class FlatSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    quenches = []
    def passthrough(atoms, surface, *, fmax, steps, terms=(), **kwargs):
        quenches.append((atoms.copy(), tuple(terms)))
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'flat')
    monkeypatch.setattr(reference, 'quench', passthrough)
    direction = np.array([[0., 0., 0.], [-1., 0., 0.]])
    mode = SimpleNamespace(direction=direction, converged=True, curvature=0.,
        residual_norm=0., force_calls=0, stop_reason='converged')
    monkeypatch.setattr(dimer, 'paper_dimer_direction', lambda *args, **kwargs: mode)
    atoms = pair()
    config = SSWConfig(width=.4, rotation_bias=2., max_gaussians=1,
        temperature_K=0., fmax=.1, relax_steps=1, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=1e-3, direction_sampling='global',
        rotation_solver='dimer')
    result = run_ssw(atoms, FlatSurface(), steps=1, config=config,
        rng=np.random.default_rng(11), move_policy=ScalarMovePolicy())
    event = result.records[0].climb[0]
    gaussian = quenches[1][1][0]
    assert event['width'] < config.width
    assert gaussian.sigma == pytest.approx(event['width'])
    np.testing.assert_allclose(gaussian.direction, event['direction'])
    assert event['protected_move']['actual_width'] == pytest.approx(gaussian.sigma)


def test_background_evaluation_failure_keeps_protected_move_diagnostics(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.dimer as dimer
    import pamssw.standalone.paper_reference as reference
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import QuenchResult

    class FailingSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            raise RuntimeError('intentional background failure')

    def passthrough(atoms, surface, *, fmax, steps, terms=(), **kwargs):
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub')
    monkeypatch.setattr(reference, 'quench', passthrough)
    direction = np.array([[0., 0., 0.], [0., 1., 0.]])
    mode = SimpleNamespace(direction=direction, converged=True, curvature=0.,
        residual_norm=0., force_calls=0, stop_reason='converged')
    monkeypatch.setattr(dimer, 'paper_dimer_direction', lambda *args, **kwargs: mode)
    config = SSWConfig(width=.2, rotation_bias=2., max_gaussians=1,
        temperature_K=0., fmax=.1, relax_steps=1, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=1e-3, direction_sampling='global',
        rotation_solver='dimer')
    result = run_ssw(pair(), FailingSurface(), steps=1, config=config,
        rng=np.random.default_rng(11), move_policy=ScalarMovePolicy())
    assert result.status == 'evaluation_failed'
    event = result.records[0].climb[0]
    assert event['protected_move']['actual_width'] == pytest.approx(.2)
    assert event['requested_width'] == pytest.approx(.2)
    np.testing.assert_array_equal(event['requested_direction'], direction)
    assert 'intentional background failure' in event['error']


def test_later_exhaustion_restores_previous_saved_gaussian_center(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.dimer as dimer
    import pamssw.standalone.paper_reference as reference
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import QuenchResult

    class FlatSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    quenches = []
    def passthrough(atoms, surface, *, fmax, steps, terms=(), **kwargs):
        quenches.append((atoms.copy(), tuple(terms)))
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'flat')
    monkeypatch.setattr(reference, 'quench', passthrough)
    directions = iter((np.array([[0., 1., 0.], [0., -1., 0.]]) / np.sqrt(2.),
                       np.array([[0., 0., 0.], [-1., 0., 0.]])))
    def solve(*args, **kwargs):
        return SimpleNamespace(direction=next(directions), converged=True,
            curvature=0., residual_norm=0., force_calls=0, stop_reason='converged')
    monkeypatch.setattr(dimer, 'paper_dimer_direction', solve)
    atoms = pair()
    config = SSWConfig(width=.2, rotation_bias=2., max_gaussians=2,
        temperature_K=0., fmax=.1, relax_steps=1, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=1e-3, direction_sampling='global',
        rotation_solver='dimer')
    result = run_ssw(atoms, FlatSurface(), steps=1, config=config,
        rng=np.random.default_rng(14), move_policy=ScalarMovePolicy(
            displacement_max=.15, max_trials=1))
    record = result.records[0]
    assert record.status == 'move_retry_exhausted'
    assert len(record.climb) == 2
    assert len(quenches) == 3  # initial, first biased stage, true landing
    np.testing.assert_allclose(quenches[-1][0].positions, atoms.positions)
    assert quenches[-1][1] == ()
    np.testing.assert_allclose(record.landing.atoms.positions, atoms.positions)


def test_move_policy_none_matches_omitted_argument(monkeypatch):
    from types import SimpleNamespace
    from ase.calculators.emt import EMT
    import pamssw.standalone.dimer as dimer
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import ASESurface

    mode = SimpleNamespace(direction=np.array([[0., 0., 0.], [0., 1., 0.]]),
        converged=True, curvature=0., residual_norm=0., force_calls=0,
        stop_reason='converged')
    monkeypatch.setattr(dimer, 'paper_dimer_direction', lambda *args, **kwargs: mode)
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=0., fmax=1e9, relax_steps=2, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=1e-3, direction_sampling='global',
        rotation_solver='dimer')
    omitted = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
        rng=np.random.default_rng(20))
    explicit = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
        rng=np.random.default_rng(20), move_policy=None)
    assert omitted.status == explicit.status
    assert omitted.evaluation_requests == explicit.evaluation_requests
    assert omitted.records[0].status == explicit.records[0].status
    np.testing.assert_array_equal(omitted.current.positions, explicit.current.positions)


@pytest.mark.parametrize('extra', [
    {'ls': object()}, {'height_policy': object()}, {'gaussian_policy': object()},
    {'reconnect_distance': 1.}, {'starter_selector': lambda *_: None},
])
def test_unsupported_policy_combinations_fail_before_pes(extra):
    from ase.calculators.emt import EMT
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import ASESurface

    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e9, relax_steps=2, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=1e-3, direction_sampling='global')
    surface = ASESurface(EMT())
    with pytest.raises(ValueError, match='protected moves cannot be combined'):
        run_ssw(atoms, surface, steps=0, config=config,
            rng=np.random.default_rng(20), move_policy=ScalarMovePolicy(), **extra)
    assert surface.requests == 0


def test_constraints_are_rejected_before_pes():
    from ase.constraints import FixAtoms
    from ase.calculators.emt import EMT
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import ASESurface

    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e9, relax_steps=2, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=1e-3, direction_sampling='global')
    surface = ASESurface(EMT())
    with pytest.raises(NotImplementedError, match='unconstrained'):
        run_ssw(atoms, surface, steps=0, config=config,
            rng=np.random.default_rng(20), move_policy=ScalarMovePolicy())
    assert surface.requests == 0
