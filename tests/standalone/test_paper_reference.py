import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']
    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = dict(energy=float((atoms.positions**2).sum()/2),
                            forces=-atoms.positions.copy())


def configuration():
    from pamssw.standalone.paper_reference import SSWConfig
    return SSWConfig(width=.2, rotation_bias=2., max_gaussians=2,
                     temperature_K=300., fmax=1e-4, relax_steps=100,
                     fd_step=.001, rotation_hvp=8, rotation_tol=1e-5,
                     direction_sampling='global')


def test_complete_walk_removes_bias_and_keeps_caller_owned_geometry():
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    atoms = Atoms('H', positions=[[.1, .2, .3]])
    original = atoms.positions.copy()
    result = run_ssw(atoms, ASESurface(Harmonic()), steps=2,
                      config=configuration(), rng=np.random.default_rng(9))
    assert len(result.records) == 2
    assert len(result.minima) == 3
    assert all(m.surface == 'true' and m.converged for m in result.minima)
    assert np.linalg.norm(result.current.positions) < 1e-4
    assert np.array_equal(atoms.positions, original)
    assert atoms.calc is None and result.current.calc is None
    assert all(len(record.climb) == 2 for record in result.records)
    assert result.evaluation_requests > 0


def test_fixed_cell_outer_escape_reuses_anchor_and_accumulates_gaussians(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import ASESurface, QuenchResult

    anchors = []
    oracle_energies = []
    oracle_forces = []
    oracle_positions = []
    biased_terms = []

    def direction_oracle(atoms, anchor, *, evaluate, **kwargs):
        anchors.append(np.array(anchor, copy=True))
        oracle_positions.append(atoms.positions.copy())
        energy, forces = evaluate(atoms)
        oracle_energies.append(energy)
        oracle_forces.append(forces.copy())
        return SimpleNamespace(direction=np.array([[1., 0., 0.]]),
                               converged=True, curvature=-1.,
                               residual_norm=0., force_calls=1)

    def biased_quench(atoms, surface, **kwargs):
        biased_terms.append(tuple(kwargs['terms']))
        return paper.BiasStageQuenchOutcome(
            QuenchResult(atoms.copy(), 1., 0., True, 0, 0, 'biased'),
            stage_stopped=False, release_all=False, diagnostics={})

    def fake_quench(atoms, surface, *, terms=(), **kwargs):
        if terms:
            raise AssertionError('biased stages must use the adapter')
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'true')

    monkeypatch.setattr(paper, 'paper_biased_direction', direction_oracle)
    monkeypatch.setattr(paper, 'quench', fake_quench)
    atoms = Atoms('H', positions=[[.1, .2, .3]])
    surface = ASESurface(Harmonic())
    result = paper.run_ssw(atoms, surface, steps=1, config=configuration(),
                           rng=np.random.default_rng(9),
                           bias_quench_adapter=biased_quench)

    assert result.records[0].status == 'gaussian_limit'
    assert len(anchors) == 2
    np.testing.assert_array_equal(anchors[0], anchors[1])
    expected_oracle_energies = [.5 * np.square(position).sum()
                                for position in oracle_positions]
    np.testing.assert_allclose(oracle_energies, expected_oracle_energies)
    np.testing.assert_allclose(oracle_forces, -np.asarray(oracle_positions))
    assert [len(terms) for terms in biased_terms] == [1, 2]
    assert biased_terms[1][0] is biased_terms[0][0]
    # Excluding the previous Gaussian must be an observable distinction.
    probe = atoms.copy()
    probe.positions = oracle_positions[1]
    previous_energy, previous_force = biased_terms[0][0].evaluate(probe)
    assert previous_energy > 1e-6
    assert np.linalg.norm(previous_force) > 1e-6


def test_budget_failure_does_not_enter_minima_archive():
    from dataclasses import replace
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    result = run_ssw(Atoms('H', positions=[[0., 0., 0.]]), ASESurface(Harmonic()),
                    steps=1, config=replace(configuration(), relax_steps=0),
                    rng=np.random.default_rng(9))
    assert len(result.minima) == 1
    assert not result.records[0].accepted
    assert result.records[0].status == 'biased_quench_failed'


def test_paper_direction_requires_nonlocal_pair_and_is_seed_reproducible():
    import pytest
    from pamssw.standalone.paper_reference import sample_initial_direction
    small = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match='more than 3'):
        sample_initial_direction(small, np.random.default_rng(1), mode='paper')
    small.positions[1, 0] = 4.
    a = sample_initial_direction(small, np.random.default_rng(1), mode='paper')
    b = sample_initial_direction(small, np.random.default_rng(1), mode='paper')
    assert np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.)
    assert a[0, 0] > 0 and a[1, 0] < 0


def test_reconnect_default_does_not_call_geometry_component(monkeypatch):
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    def forbidden(*args, **kwargs):
        raise AssertionError('default SSW must not invoke reconnection')
    monkeypatch.setattr('pamssw.standalone.paper_reference.reconnect_clusters', forbidden)
    run_ssw(Atoms('H', positions=[[.1, .2, .3]]), ASESurface(Harmonic()), steps=1,
            config=configuration(), rng=np.random.default_rng(9))


def test_reconnect_distance_rejects_periodic_before_first_pes_request():
    import pytest
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    atoms = Atoms('H2', positions=[[0., 0., 0.], [4., 0., 0.]],
                  cell=[10., 10., 10.], pbc=True)
    surface = ASESurface(Harmonic())
    config = configuration()
    with pytest.raises(ValueError, match='nonperiodic'):
        run_ssw(atoms, surface, steps=0, config=config,
                rng=np.random.default_rng(9), reconnect_distance=1.7)
    assert surface.requests == 0


def test_reconnect_step_records_result_and_quenches_copy(monkeypatch):
    from dataclasses import replace
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    from pamssw.standalone.cluster_reconnection import reconnect_clusters as real_reconnect
    calls = []
    quench_inputs = []
    original_quench = __import__('pamssw.standalone.paper_reference', fromlist=['quench']).quench
    def observed_quench(*args, **kwargs):
        quench_inputs.append((args[0].positions.copy(), kwargs.get('terms')))
        return original_quench(*args, **kwargs)
    monkeypatch.setattr('pamssw.standalone.paper_reference.quench', observed_quench)
    def wrapped(atoms, criterion, **kwargs):
        calls.append((kwargs.get('repair'), atoms.positions.copy()))
        result = real_reconnect(atoms, criterion, **kwargs)
        if kwargs.get('repair'):
            moved = result.atoms.copy()
            moved.positions[-1, 0] += 2.0
            return replace(result, atoms=moved)
        return result
    monkeypatch.setattr('pamssw.standalone.paper_reference.reconnect_clusters', wrapped)
    atoms = Atoms('H2', positions=[[0., 0., 0.], [4., 0., 0.]])
    result = run_ssw(atoms, ASESurface(Harmonic()), steps=1,
                     config=configuration(), rng=np.random.default_rng(9),
                     reconnect_distance=1.7)
    assert calls and calls[0][0] is False and calls[-1][0] is True
    assert result.records[0].cluster_reconnection is not None
    assert result.records[0].cluster_reconnection.atoms is not result.records[0].last_atoms
    assert np.allclose(quench_inputs[-1][0], result.records[0].cluster_reconnection.atoms.positions)
    assert quench_inputs[-1][1] is None
    assert not np.allclose(quench_inputs[-1][0], result.records[0].last_atoms.positions)


def test_bias_adapter_rejects_bad_return_before_pes():
    import pytest
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    atoms = Atoms('H', positions=[[.1, .2, .3]])
    surface = ASESurface(Harmonic())
    with pytest.raises(TypeError, match='bias_quench_adapter'):
        run_ssw(atoms, surface, steps=0, config=configuration(),
                rng=np.random.default_rng(9), bias_quench_adapter=object())
    assert surface.requests == 0


def test_bias_adapter_stage_stop_can_continue_and_release_is_explicit(monkeypatch):
    from dataclasses import replace
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import ASESurface
    seen = []
    original = paper.quench

    def adapter(atoms, surface, **kwargs):
        callback_context = kwargs.pop('context')
        seen.append((atoms.copy(), callback_context))
        assert callback_context['outer_index'] == 0
        relaxed = original(atoms, surface, **kwargs)
        return paper.BiasStageQuenchOutcome(relaxed, stage_stopped=True,
                                            release_all=False, diagnostics={'tag': 'test'})

    cfg = replace(configuration(), max_gaussians=1)
    result = paper.run_ssw(Atoms('H', positions=[[.1, .2, .3]]), ASESurface(Harmonic()),
                           steps=1, config=cfg, rng=np.random.default_rng(9),
                           bias_quench_adapter=adapter)
    assert seen and set(seen[0][1]) == {
        'current', 'current_energy', 'best_energy', 'outer_index',
        'gaussian_index', 'center', 'soft_terms', 'max_gaussians'}
    assert result.records[0].climb[0]['stage_stop_reason'] == 'adapter'
    assert result.records[0].climb[0]['diagnostics'] == {'tag': 'test'}
    assert result.records[0].status == 'gaussian_limit'
    assert seen[0][1]['soft_terms'] == ()


def test_bias_adapter_stage_stop_reaches_next_gaussian(monkeypatch):
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import ASESurface
    original = paper.quench
    calls = []

    def adapter(atoms, surface, **kwargs):
        calls.append(kwargs['context']['gaussian_index'])
        kwargs.pop('context')
        return paper.BiasStageQuenchOutcome(original(atoms, surface, **kwargs), True, False, {})

    cfg = configuration()
    result = paper.run_ssw(Atoms('H', positions=[[.1, .2, .3]]), ASESurface(Harmonic()),
                           steps=1, config=cfg, rng=np.random.default_rng(9),
                           bias_quench_adapter=adapter)
    assert calls == [0, 1]
    assert [e['status'] for e in result.records[0].climb] == ['stage_stopped', 'stage_stopped']


def test_bias_adapter_release_enters_stage_release_landing(monkeypatch):
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import ASESurface
    original = paper.quench

    def adapter(atoms, surface, **kwargs):
        kwargs.pop('context')
        return paper.BiasStageQuenchOutcome(
            original(atoms, surface, **kwargs), stage_stopped=True,
            release_all=True, diagnostics={})

    result = paper.run_ssw(Atoms('H', positions=[[.1, .2, .3]]), ASESurface(Harmonic()),
                           steps=1, config=configuration(), rng=np.random.default_rng(9),
                           bias_quench_adapter=adapter)
    assert result.records[0].status == 'stage_release'
    assert len(result.records[0].climb) == 1
    assert result.records[0].landing is not None
    assert result.records[0].landing.surface == 'true'


def test_bias_adapter_nonfinite_result_is_not_archived():
    import pytest
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import ASESurface, QuenchResult

    def adapter(atoms, surface, **kwargs):
        kwargs.pop('context')
        return paper.BiasStageQuenchOutcome(
            QuenchResult(atoms.copy(), np.nan, 1., False, 0, 0, 'biased'),
            stage_stopped=True, release_all=False, diagnostics={})

    result = paper.run_ssw(Atoms('H', positions=[[.1, .2, .3]]), ASESurface(Harmonic()),
                           steps=1, config=configuration(), rng=np.random.default_rng(9),
                           bias_quench_adapter=adapter)
    assert len(result.minima) == 1
    assert result.records[0].status == 'evaluation_failed'


def test_bias_adapter_and_checkpoint_are_rejected_before_pes(tmp_path):
    import pytest
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import ASESurface
    atoms = Atoms('H', positions=[[.1, .2, .3]])
    surface = ASESurface(Harmonic())
    adapter = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match='checkpoint'):
        paper.run_ssw(atoms, surface, steps=0, config=configuration(),
                      rng=np.random.default_rng(9), bias_quench_adapter=adapter,
                      checkpoint_path=tmp_path / 'cp.pkl')
    assert surface.requests == 0


def test_bias_adapter_noop_preserves_default_oracle_trace():
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import ASESurface
    plain_surface = ASESurface(Harmonic())
    adapted_surface = ASESurface(Harmonic())
    original = paper.quench

    def adapter(atoms, surface, **kwargs):
        kwargs.pop('context')
        return paper.BiasStageQuenchOutcome(original(atoms, surface, **kwargs), False, False, {})

    plain = paper.run_ssw(Atoms('H', positions=[[.1, .2, .3]]), plain_surface,
                         steps=1, config=configuration(), rng=np.random.default_rng(9))
    adapted = paper.run_ssw(Atoms('H', positions=[[.1, .2, .3]]), adapted_surface,
                            steps=1, config=configuration(), rng=np.random.default_rng(9),
                            bias_quench_adapter=adapter)
    assert plain_surface.requests == adapted_surface.requests
    assert np.array_equal(plain.current.positions, adapted.current.positions)
    assert np.array_equal(plain.records[0].last_atoms.positions, adapted.records[0].last_atoms.positions)


def test_run_ls_ssw_forwards_reconnect_keyword(monkeypatch):
    import pamssw.standalone.paper_reference as paper
    seen = {}
    sentinel = object()
    def fake_run_ssw(*args, **kwargs):
        seen.update(kwargs)
        return sentinel
    monkeypatch.setattr(paper, 'run_ssw', fake_run_ssw)
    ls = paper.LSSettings({}, {}, 0.0)
    assert paper.run_ls_ssw('atoms', 'surface', steps=0, config='config',
                            rng='rng', ls=ls, reconnect_distance=1.7) is sentinel
    assert seen['reconnect_distance'] == 1.7
