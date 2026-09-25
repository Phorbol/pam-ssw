import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from pamssw.standalone.paper_reference import SSWProgress, load_ssw_checkpoint, run_ssw
from pamssw.standalone.surface import ASESurface, quench
from test_ssw_checkpoint import _case


def test_progress_pause_at_call_start_returns_detached_initial_checkpoint():
    atoms, config, ls = _case()
    surface = ASESurface(EMT())
    seen = []

    def pause(progress):
        seen.append(progress)
        assert progress.kind == 'initial'
        assert progress.step is None
        assert progress.new_minimum.energy == progress.current_energy
        progress.current.positions[:] = 1000.0
        progress.new_minimum.atoms.positions[:] = 1000.0
        progress.best.atoms.positions[:] = 1000.0
        return True

    result = run_ssw(atoms, surface, steps=2, config=config,
                     rng=np.random.default_rng(19), ls=ls,
                     progress_callback=pause)
    assert result.status == 'paused'
    assert result.checkpoint.status == 'completed'
    assert result.checkpoint.next_index == 0
    assert result.records == ()
    assert result.evaluation_requests == surface.requests
    assert seen[0].evaluation_requests == surface.requests
    np.testing.assert_array_equal(result.current.positions,
                                  result.checkpoint.current.positions)
    assert not np.all(result.current.positions == 1000.0)


def test_progress_outer_pause_and_resume_preserve_ls_rng_and_history():
    atoms, config, ls = _case()
    full_rng = np.random.default_rng(19)
    full = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                   rng=full_rng, ls=ls)
    seen = []

    def pause(progress):
        seen.append(progress)
        if progress.kind == 'initial':
            assert progress.new_minimum is not None
            return False
        assert progress.kind == 'outer_step'
        assert progress.step.index == 0
        assert progress.new_minimum is None or progress.new_minimum.converged
        progress.current.positions[:] = 1000.0
        progress.best.atoms.positions[:] = 1000.0
        if progress.step.last_atoms is not None:
            progress.step.last_atoms.positions[:] = 1000.0
        return True

    first_surface = ASESurface(EMT())
    first = run_ssw(atoms, first_surface, steps=2, config=config,
                    rng=np.random.default_rng(19), ls=ls,
                    progress_callback=pause)
    assert first.status == 'paused'
    assert first.checkpoint.next_index == 1
    assert len(first.checkpoint.records) == len(first.records) == 1
    assert first.evaluation_requests == first_surface.requests
    assert seen[-1].evaluation_requests == first.evaluation_requests
    assert not np.all(first.current.positions == 1000.0)
    assert seen[0].next_index == 0

    resumed_rng = np.random.default_rng(999)
    resumed_surface = ASESurface(EMT())
    resumed_start = []
    resumed = run_ssw(atoms, resumed_surface, steps=1, config=config,
                      rng=resumed_rng, ls=ls, checkpoint=first.checkpoint,
                      progress_callback=lambda p: resumed_start.append(p) or False)
    assert resumed_start[0].kind == 'initial'
    assert resumed_start[0].new_minimum is None
    assert resumed_start[0].next_index == 1
    assert resumed.evaluation_requests == full.evaluation_requests
    assert first.evaluation_requests + resumed_surface.requests == full.evaluation_requests
    assert [r.accepted for r in resumed.records] == [r.accepted for r in full.records]
    assert [r.ls_update for r in resumed.records] == [r.ls_update for r in full.records]
    np.testing.assert_array_equal(resumed.current.positions, full.current.positions)
    assert resumed_rng.bit_generator.state == full_rng.bit_generator.state


def test_progress_zero_outer_steps_still_reports_call_start_and_final_checkpoint():
    atoms, config, ls = _case()
    seen = []
    result = run_ssw(atoms, ASESurface(EMT()), steps=0, config=config,
                     rng=np.random.default_rng(19), ls=ls,
                     progress_callback=lambda p: seen.append(p) or False)
    assert result.status == 'completed'
    assert len(result.records) == 0
    assert result.checkpoint.next_index == 0
    assert [p.kind for p in seen] == ['initial']
    assert seen[0].new_minimum is not None


def test_progress_terminal_outer_failure_returns_paid_terminal_checkpoint():
    atoms, config, ls = _case()
    seed = quench(atoms, ASESurface(EMT()), fmax=config.fmax,
                  steps=config.relax_steps, optimizer=LBFGS,
                  lbfgs_memory=config.lbfgs_memory)

    class BudgetSurface(ASESurface):
        def evaluate(self, candidate):
            if self.requests >= seed.evaluation_requests:
                raise RuntimeError('intentional terminal request cap')
            return super().evaluate(candidate)

    surface = BudgetSurface(EMT())
    seen = []
    result = run_ssw(atoms, surface, steps=2, config=config,
                     rng=np.random.default_rng(19),
                     progress_callback=lambda p: seen.append(p) or False)
    assert result.status == result.checkpoint.status == 'evaluation_failed'
    assert result.evaluation_requests == result.checkpoint.evaluation_requests == surface.requests
    assert result.checkpoint.next_index == 1
    assert [p.kind for p in seen] == ['initial']
    assert 'intentional terminal request cap' in result.records[-1].error


def test_ls_initialization_failure_returns_diagnostic_checkpoint_without_event(monkeypatch):
    import pamssw.standalone.paper_reference as ref

    atoms, config, ls = _case()
    surface = ASESurface(EMT())
    seen = []

    def fail_ls_initialization(*_args, **_kwargs):
        raise ValueError('intentional LS initialization failure')

    monkeypatch.setattr(ref, '_initialize_ls_state', fail_ls_initialization)
    result = run_ssw(atoms, surface, steps=2, config=config,
                     rng=np.random.default_rng(19), ls=ls,
                     progress_callback=lambda progress: seen.append(progress) or True)
    assert result.status == result.checkpoint.status == 'ls_initialization_failed'
    assert result.checkpoint.next_index == 0
    assert result.evaluation_requests == result.checkpoint.evaluation_requests == surface.requests
    assert result.records[0].status == 'ls_initialization_failed'
    assert seen == []


def test_progress_only_copies_full_history_once_at_return(monkeypatch):
    import pamssw.standalone.paper_reference as ref

    atoms, config, ls = _case()
    first = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                    rng=np.random.default_rng(19), ls=ls,
                    progress_callback=lambda _p: False)
    copied_record_lists = []
    original = ref._checkpoint_copy

    def track(value):
        if isinstance(value, list) and value and isinstance(value[0], ref.SSWStep):
            copied_record_lists.append(len(value))
        return original(value)

    monkeypatch.setattr(ref, '_checkpoint_copy', track)
    result = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                     rng=np.random.default_rng(999), ls=ls, checkpoint=first.checkpoint,
                     progress_callback=lambda _p: False)
    assert result.status == 'completed'
    assert len(result.records) == 3
    assert copied_record_lists == [3]


def test_progress_with_checkpoint_path_keeps_each_outer_persistence(tmp_path):
    atoms, config, ls = _case()
    path = tmp_path / 'progress-path.pkl'
    seen = []
    result = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                     rng=np.random.default_rng(19), ls=ls,
                     checkpoint_path=path,
                     progress_callback=lambda p: seen.append(p) or p.kind == 'outer_step')
    stored = load_ssw_checkpoint(path)
    assert result.status == 'paused'
    assert result.checkpoint.next_index == stored.next_index == 1
    assert result.checkpoint.evaluation_requests == stored.evaluation_requests
    assert [p.kind for p in seen] == ['initial', 'outer_step']


@pytest.mark.parametrize('steps', [0, 2])
def test_resumed_initial_pause_with_checkpoint_path_persists_incoming_checkpoint(
        tmp_path, monkeypatch, steps):
    import pamssw.standalone.paper_reference as ref

    atoms, config, ls = _case()
    first = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                    rng=np.random.default_rng(19), ls=ls,
                    progress_callback=lambda _p: False)
    path = tmp_path / 'initial-pause-resume.pkl'
    seen = []
    writes = []
    original_save = ref.save_ssw_checkpoint

    def track_save(destination, checkpoint):
        writes.append(destination)
        return original_save(destination, checkpoint)

    monkeypatch.setattr(ref, 'save_ssw_checkpoint', track_save)
    result = run_ssw(atoms, ASESurface(EMT()), steps=steps, config=config,
                     rng=np.random.default_rng(999), ls=ls,
                     checkpoint=first.checkpoint, checkpoint_path=path,
                     progress_callback=lambda p: seen.append(p) or True)
    stored = load_ssw_checkpoint(path)
    assert result.status == 'paused'
    assert [p.kind for p in seen] == ['initial']
    assert writes == [path]
    assert stored.next_index == first.checkpoint.next_index == result.checkpoint.next_index
    assert stored.evaluation_requests == first.checkpoint.evaluation_requests
    np.testing.assert_array_equal(stored.current.positions,
                                  first.checkpoint.current.positions)


def test_ls_prequench_failure_progress_only_returns_terminal_checkpoint(monkeypatch):
    import pamssw.standalone.paper_reference as ref

    atoms, config, ls = _case()
    surface = ASESurface(EMT())

    def fail_prequench(*_args, **_kwargs):
        raise RuntimeError('intentional LS prequench failure')

    monkeypatch.setattr(ref, 'prepare_ls_step', fail_prequench)
    seen = []
    result = run_ssw(atoms, surface, steps=2, config=config,
                     rng=np.random.default_rng(19), ls=ls,
                     progress_callback=lambda p: seen.append(p) or False)
    assert result.status == result.checkpoint.status == 'ls_prequench_failed'
    assert result.checkpoint.next_index == 1
    assert len(result.records) == len(result.checkpoint.records) == 1
    assert result.records[0].status == 'ls_prequench_failed'
    assert result.evaluation_requests == result.checkpoint.evaluation_requests == surface.requests
    assert result.records[0].evaluation_requests == 0
    assert [p.kind for p in seen] == ['initial']


def test_progress_and_checkpoint_callbacks_are_mutually_exclusive_without_pes():
    atoms, config, ls = _case()
    surface = ASESurface(EMT())
    with pytest.raises(ValueError, match='mutually exclusive'):
        run_ssw(atoms, surface, steps=1, config=config,
                rng=np.random.default_rng(19), ls=ls,
                progress_callback=lambda _: False,
                checkpoint_callback=lambda _: False)
    assert surface.requests == 0


def test_progress_callback_requires_pool_checkpoint_contract_without_pes():
    atoms, config, ls = _case()
    surface = ASESurface(EMT())
    incomplete_selector = lambda _snapshot, _rng: None
    with pytest.raises((TypeError, ValueError), match='checkpoint|contract|state|restore'):
        run_ssw(atoms, surface, steps=1, config=config,
                rng=np.random.default_rng(19), ls=ls,
                starter_selector=incomplete_selector,
                selector_rng=np.random.default_rng(23),
                progress_callback=lambda _: False)
    assert surface.requests == 0
