import numpy as np
import pytest
from ase import Atoms
from ase.cluster import Icosahedron

from pamssw.standalone.descriptor_direction import (
    DescriptorMode, descriptor_value_gradient, normalized_descriptor_direction,
)
from pamssw.standalone.recovered_direction import (
    RecoveredDirectionController, RecoveredDirectionSettings,
)


def mode(family='S1', **kw):
    defaults = dict(family=family, center=0, neighbor_species=(), cutoff=3.2,
                    exponent=2, epsilon=1e-8, degree=(2 if family == 'S2' else None))
    defaults.update(kw)
    return DescriptorMode(**defaults)


def _fd(atoms, descriptor_mode, atom, axis, h=1e-6):
    plus = atoms.copy(); minus = atoms.copy()
    plus.positions[atom, axis] += h
    minus.positions[atom, axis] -= h
    return (descriptor_value_gradient(plus, descriptor_mode)[0] -
            descriptor_value_gradient(minus, descriptor_mode)[0]) / (2*h)


@pytest.mark.parametrize('family', ['S1', 'S2'])
def test_analytic_all_atom_gradient_matches_finite_difference(family):
    atoms = Atoms('COH', positions=[[0.1, 0.2, 0.3], [1.2, 0.3, 0.1],
                                    [-0.2, 1.1, 0.7]])
    descriptor_mode = mode(family, neighbor_species=(1, 8))
    value, gradient, images = descriptor_value_gradient(atoms, descriptor_mode)
    assert value > 0 and images == 2
    for atom in range(3):
        for axis in range(3):
            assert gradient[atom, axis] == pytest.approx(
                _fd(atoms, descriptor_mode, atom, axis), abs=2e-6)


def test_periodic_images_including_self_images_map_back_to_original_atoms():
    atoms = Atoms('H', positions=[[0.13, 0.27, 0.31]], cell=np.eye(3), pbc=True)
    descriptor_mode = mode(cutoff=1.01)
    value, gradient, images = descriptor_value_gradient(atoms, descriptor_mode)
    assert images >= 6
    assert value > 0
    np.testing.assert_allclose(gradient, 0., atol=1e-12)


def test_periodic_image_gradient_matches_finite_difference_across_boundary():
    atoms = Atoms('H2', positions=[[.2, .4, .5], [4.3, .4, .5]],
                  cell=[4.5, 5., 5.], pbc=True)
    descriptor_mode = mode(cutoff=1.1)
    value, gradient, images = descriptor_value_gradient(atoms, descriptor_mode)
    assert value > 0 and images == 1
    for atom in range(2):
        assert gradient[atom, 0] == pytest.approx(
            _fd(atoms, descriptor_mode, atom, 0), abs=2e-6)


def test_active_projection_and_zero_projected_gradient_are_explicit():
    atoms = Atoms('H3', positions=[[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
    active = np.array([True, False, False])
    direction, _, _, norm = normalized_descriptor_direction(atoms, mode(exponent=1), active)
    assert norm == 0.
    assert direction is None
    displaced = atoms.copy(); displaced.positions[2, 1] = .2
    projected, _, _, projected_norm = normalized_descriptor_direction(displaced, mode(), active)
    assert projected_norm > 0 and np.count_nonzero(projected[1:]) == 0


def _settings(**kw):
    values = dict(ratio_local=1, local_probability=1., group_threshold=.5,
                  pre_rotmax=1, rotmax=2, pre_ftol=1e-3, ftol=1e-3,
                  metric='euclidean', max_force_calls=4)
    values.update(kw)
    return RecoveredDirectionSettings(**values)


def test_disabled_descriptor_modes_consume_no_rng_draws():
    atoms = Icosahedron('Cu', 2)
    a = np.random.default_rng(8); b = np.random.default_rng(8)
    controller = RecoveredDirectionController(_settings())
    controller.initialize(atoms, atoms.copy(), a)
    controller.begin_escape(atoms, atoms.copy(), a)
    # The disabled selector itself does not consume a draw; pair/group and
    # established local selector draws are the only consumers on both sides.
    controller2 = RecoveredDirectionController(_settings(descriptor_modes=(mode(),), descriptor_probability=0.))
    controller2.initialize(atoms, atoms.copy(), b)
    controller2.begin_escape(atoms, atoms.copy(), b)
    np.testing.assert_array_equal(a.random(4), b.random(4))
    assert controller2.diagnostics.get('descriptor_direction') is None


def test_selected_mode_recomputes_each_gaussian_and_records_provenance():
    atoms = Icosahedron('Cu', 2)
    controller = RecoveredDirectionController(_settings(
        descriptor_modes=(mode(),), descriptor_probability=1.))
    controller.initialize(atoms, atoms.copy(), np.random.default_rng(3))
    first_atoms = atoms.copy()
    first = controller.begin_escape(atoms, first_atoms, np.random.default_rng(4))
    first_direction = first.direction.copy()
    assert not first.release_all
    changed = atoms.copy(); changed.positions[2, 2] += .2
    second = controller.update_direction(changed, np.random.default_rng(5))
    assert not np.allclose(first_direction, second.direction)
    diagnostic = controller.diagnostics['descriptor_direction']
    assert diagnostic['direction_source'] == 'explicit_descriptor_gradient'
    assert diagnostic['family'] == 'S1' and diagnostic['image_count'] > 0


def test_zero_descriptor_gradient_returns_unavailable_without_random_fallback():
    atoms = Icosahedron('Cu', 2)
    unavailable_mode = mode(neighbor_species=(6,))
    controller = RecoveredDirectionController(_settings(
        descriptor_modes=(unavailable_mode,), descriptor_probability=1.))
    controller.initialize(atoms, atoms.copy(), np.random.default_rng(31))
    result = controller.begin_escape(atoms, atoms.copy(), np.random.default_rng(32))
    assert result.release_all
    assert np.count_nonzero(result.direction) == 0
    assert controller.diagnostics['descriptor_direction']['stop_reason'] == 'zero_descriptor_gradient'


def test_descriptor_checkpoint_uses_new_wrapper_schema():
    from pamssw.standalone.paper_reference import SSWCheckpoint, SSWConfig, _validate_ssw_checkpoint
    from pamssw.standalone.recovered_direction import RecoveredDirectionCheckpointState
    from pamssw.standalone.surface import QuenchResult
    atoms = Atoms('H2', positions=[[0., 0., 0.], [1., 0., 0.]])
    settings = _settings(descriptor_modes=(mode(),), descriptor_probability=.25)
    state = RecoveredDirectionCheckpointState(settings, (0, 1), np.zeros(2, dtype=int), None)
    q = QuenchResult(atoms, 0., 0., True, 0, 0, 'stub')
    config = __import__('pamssw.standalone.paper_reference', fromlist=['SSWConfig']).SSWConfig(
        width=.1, rotation_bias=1., max_gaussians=1, temperature_K=0., fmax=.1,
        relax_steps=0, fd_step=.01, rotation_hvp=2, rotation_tol=.01)
    checkpoint = SSWCheckpoint(q, q, 0., q, (), (), None, None, config, None, None,
        None, 0, None, {}, 0, 0, 'completed', schema_version=8,
        recovered_direction_state=state, base_schema_version=4)
    _validate_ssw_checkpoint(checkpoint)
    with pytest.raises(ValueError, match='schema 8'):
        _validate_ssw_checkpoint(__import__('dataclasses').replace(checkpoint, schema_version=4,
                                                                    base_schema_version=None))


def test_periodic_descriptor_checkpoint_without_hookean_roundtrip(tmp_path):
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw, load_ssw_checkpoint

    class Flat:
        def __init__(self):
            self.requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    atoms = Atoms('Cu3', positions=[[0., 0., 0.], [1.5, .4, .2], [.3, 1.9, .7]],
                  cell=[7., 7., 7.], pbc=True)
    settings = _settings(geometry='periodic_local', descriptor_modes=(mode(),),
                         descriptor_probability=1.)
    config = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
                       temperature_K=0., fmax=.05, relax_steps=0, fd_step=.001,
                       rotation_hvp=2, rotation_tol=.02,
                       direction_sampling='global', cluster_frame='translation_only')
    path = tmp_path / 'periodic.pkl'
    first = run_ssw(atoms, Flat(), steps=0, config=config,
                    rng=np.random.default_rng(31), recovered_direction=settings,
                    checkpoint_path=path)
    cp = load_ssw_checkpoint(path)
    assert cp.schema_version == 8 and not cp.hookean_specs
    resumed = run_ssw(atoms, Flat(), steps=0, config=config,
                      rng=np.random.default_rng(99), checkpoint=cp)
    np.testing.assert_array_equal(resumed.current.positions, first.current.positions)
    assert resumed.checkpoint.rng_state == first.checkpoint.rng_state


def test_recovered_direction_checkpoint_binds_the_mode_library():
    atoms = Icosahedron('Cu', 2)
    first = RecoveredDirectionController(_settings(
        descriptor_modes=(mode(cutoff=2.8),), descriptor_probability=.4))
    first.initialize(atoms, atoms.copy(), np.random.default_rng(17))
    state = first.checkpoint_state()
    changed = RecoveredDirectionController(_settings(
        descriptor_modes=(mode(cutoff=2.9),), descriptor_probability=.4))
    with pytest.raises(ValueError, match='settings'):
        changed.restore_checkpoint_state(state)


def test_q_unavailable_candidate_does_not_run_true_landing_quench(monkeypatch):
    import pamssw.standalone.paper_reference as driver
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import QuenchResult

    class FlatSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    atoms = Atoms('H5', positions=[[0., 0., 0.], [1., .2, 0.], [0., 2., .1],
                                   [2.5, 1., .4], [-2., 1.1, .2]])
    calls = []
    def fake_quench(atoms, surface, **kwargs):
        calls.append(1)
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub')
    monkeypatch.setattr(driver, 'quench', fake_quench)
    settings = _settings(descriptor_modes=(mode(neighbor_species=(6,)),),
                         descriptor_probability=1.)
    config = SSWConfig(width=.1, rotation_bias=1., max_gaussians=2,
        temperature_K=0., fmax=.1, relax_steps=0, fd_step=.01,
        rotation_hvp=2, rotation_tol=.01, cluster_frame='direction_only')
    result = run_ssw(atoms, FlatSurface(), steps=1, config=config,
                     rng=np.random.default_rng(11), recovered_direction=settings)
    assert len(calls) == 1  # Initial minimum only; no true landing quench.
    assert result.records[0].status == 'descriptor_direction_unavailable'
    assert result.records[0].climb[0]['recovered_direction']['descriptor_direction']['stop_reason'] == 'zero_descriptor_gradient'


def test_constrained_q_unavailable_candidate_does_not_run_true_landing_quench(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.constrained_reference as driver
    from pamssw.standalone.constrained_reference import ConstrainedSSWConfig, run_constrained_ssw

    class FlatSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    atoms = Icosahedron('Cu', 2)
    calls = []
    def fake_quench(atoms, surface, **kwargs):
        calls.append(1)
        return driver.ConstrainedQuenchResult(atoms.copy(), 0., 0., 0.,
            SimpleNamespace(converged=True), {'certified': True}, 0)
    monkeypatch.setattr(driver, 'constrained_quench', fake_quench)
    settings = _settings(descriptor_modes=(mode(neighbor_species=(6,)),),
                         descriptor_probability=1.)
    config = ConstrainedSSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
                                  relax_steps=2, fmax=.1)
    result = run_constrained_ssw(atoms, FlatSurface(), steps=1, config=config,
        rng=np.random.default_rng(13), fixed_indices=[0], recovered_direction=settings)
    assert len(calls) == 1  # Initial minimum only; no true landing quench.
    record = result.records[-1]
    assert record['status'] == 'descriptor_direction_unavailable'
    assert record['climb'][0]['recovered_direction']['descriptor_direction']['stop_reason'] == 'zero_descriptor_gradient'


def test_q_checkpoint_split_resume_matches_continuous_and_binds_rng(monkeypatch, tmp_path):
    from ase.cluster import Icosahedron
    import pamssw.standalone.paper_reference as driver
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw, load_ssw_checkpoint
    from pamssw.standalone.surface import QuenchResult

    class FlatSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kwargs:
        QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub'))
    # This is a state/restart test, not a flat-PES CBD test. Exact flat
    # forces make the native acos rotation undefined. Isolate the already
    # separately tested CBD while exercising actual descriptor regeneration.
    import pamssw.standalone.recovered_cbd as cbd
    def identity_rotation(atoms, anchor, **kwargs):
        return cbd.RecoveredCBDResult(anchor.copy(), 0., 0., 0., 0,
            'CBD_unbiasedRot', True, True, 'force_converged', 0., (), anchor.copy())
    monkeypatch.setattr(cbd, 'recovered_cbd_direction', identity_rotation)
    atoms = Icosahedron('Cu', 2)
    settings = _settings(descriptor_modes=(mode(cutoff=4.2),), descriptor_probability=1.)
    config = SSWConfig(width=.1, rotation_bias=1., max_gaussians=2,
        temperature_K=0., fmax=.03, relax_steps=0, fd_step=.001,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global',
        cluster_frame='direction_only')
    continuous = run_ssw(atoms.copy(), FlatSurface(), steps=2, config=config,
        rng=np.random.default_rng(19), recovered_direction=settings)
    assert continuous.status == 'completed', [(r.status, r.error) for r in continuous.records]
    path = tmp_path / 'q-state.pkl'
    run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config,
        rng=np.random.default_rng(19), recovered_direction=settings, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert checkpoint.schema_version == 8
    from dataclasses import replace
    downgraded_surface = FlatSurface()
    with pytest.raises(ValueError, match='schema 8'):
        run_ssw(atoms.copy(), downgraded_surface, steps=0, config=config,
            rng=np.random.default_rng(999), checkpoint=replace(checkpoint, schema_version=4, base_schema_version=None))
    assert downgraded_surface.requests == 0
    changed_surface = FlatSurface()
    changed_settings = _settings(descriptor_modes=(mode(cutoff=4.3),), descriptor_probability=1.)
    with pytest.raises(ValueError, match='settings'):
        run_ssw(atoms.copy(), changed_surface, steps=0, config=config,
            rng=np.random.default_rng(999), recovered_direction=changed_settings,
            checkpoint=checkpoint)
    assert changed_surface.requests == 0
    resumed = run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config,
        rng=np.random.default_rng(999), recovered_direction=settings, checkpoint=checkpoint)
    first = continuous.records[1].climb[0]['recovered_direction']
    second = resumed.records[1].climb[0]['recovered_direction']
    assert first['descriptor_direction'] == second['descriptor_direction']
    np.testing.assert_allclose(first['proposal'], second['proposal'])
