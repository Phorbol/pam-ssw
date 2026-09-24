"""Caller budget/segmentation regressions; not evidence of C60 search quality."""
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pytest
from ase.io import write
from pamssw.standalone import load_ssw_checkpoint
from test_ssw_checkpoint import _case

spec = importlib.util.spec_from_file_location('c60long', Path(__file__).parents[2] / 'research/ga_ssw/c60_long_budget.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def prepare(folder, native=False):
    folder.mkdir()
    atoms, config, _ = _case()
    config = replace(config, quench_optimizer="safe-lbfgs-total", lbfgs_memory=500)
    write(folder / 'input.traj', atoms)
    ls = dict(bond_energies={'29,29': 1.}, bond_lengths={'29,29': 3.},
              target_mev_per_atom=1., prequench={'fmax': .1, 'steps': 50,
                                              'exit_policy': 'force_or_step_limit'}) if native else None
    plan = dict(seed=19, backend='emt', input='input.traj', input_sha256=r.sha(folder / 'input.traj'),
        ssw_config=asdict(config), native_ls=ls,
        native_mc={'energy_tol_eV': .1, 'maxtrap': 99999},
        recovered_rotation={'pre_rotmax': 5, 'rotmax': 15, 'pre_ftol': .2,
                            'ftol': .02, 'metric': 'euclidean', 'max_force_calls': 40},
        search_cap=10000, fresh_cap=3, wall_seconds=3600)
    r.atomic_json(folder / 'plan.json', plan)
    return plan


@pytest.mark.parametrize('native', [False, True])
def test_segmented_state_and_cost_match_continuous(tmp_path, native):
    a, b = tmp_path / 'full', tmp_path / 'split'
    prepare(a, native); prepare(b, native)
    full = r.run_segment(a, 900, max_attempts=3)
    first = r.run_segment(b, 900, max_attempts=1)
    split = r.run_segment(b, 900, max_attempts=2)
    x, y = (load_ssw_checkpoint(p / 'checkpoint.pkl') for p in (a, b))
    assert full['search'] == split['search'] == x.evaluation_requests == y.evaluation_requests
    assert 0 < first['search'] < split['search']
    assert x.rng_state == y.rng_state
    assert x.next_index == y.next_index == 3
    np.testing.assert_array_equal(x.current.positions, y.current.positions)
    assert [(t.status, t.accepted, t.ls_update) for t in x.records] == [
        (t.status, t.accepted, t.ls_update) for t in y.records]
    if native:
        assert vars(x.response) == vars(y.response)
    assert split['reserved_seconds'] == 1800
    assert all(0 <= item['budget_io_seconds'] <= item['elapsed_seconds'] for item in split['segments'])


def test_durable_charge_never_refunded_and_caps(tmp_path):
    plan = prepare(tmp_path / 'arm')
    folder = tmp_path / 'arm'
    plan.update(search_cap=2, fresh_cap=1, wall_seconds=900)
    r.atomic_json(folder / 'plan.json', plan)
    budget = r.Budget(folder, plan); budget.begin(400)
    assert budget.charge('search') == 1
    recovered = r.Budget(folder, plan)
    assert recovered.state['search'] == 2  # unfinished block is conservatively consumed
    with pytest.raises(ValueError, match='interrupted'):
        recovered.begin(400)
    # Budget is already exhausted by the conservative recovery.
    with pytest.raises(ValueError, match='exhausted'):
        recovered.begin(400, interrupted=True)
    with pytest.raises(RuntimeError, match='search_budget'):
        recovered.charge('search')
    assert recovered.charge('fresh') == 1
    with pytest.raises(RuntimeError, match='fresh_budget'):
        recovered.charge('fresh')
    with pytest.raises(ValueError, match='allocation'):
        recovered.begin(501, interrupted=True)
    assert recovered.state['reserved_seconds'] == 400


def test_changed_plan_rejected_before_calculator(tmp_path):
    folder = tmp_path / 'arm'; plan = prepare(folder)
    r.Budget(folder, plan)
    plan['seed'] = 20; r.atomic_json(folder / 'plan.json', plan)
    with pytest.raises(ValueError, match='plan changed'):
        r.Budget(folder, plan)


def test_interrupted_paid_request_survives_checkpoint_rollback(tmp_path):
    folder = tmp_path / 'arm'; plan = prepare(folder)
    r.run_segment(folder, 600, max_attempts=1)
    saved = load_ssw_checkpoint(folder / 'checkpoint.pkl')
    budget = r.Budget(folder, plan); budget.begin(600)
    budget.charge('search')  # emulate process death after charge but before evaluation
    state = r.run_segment(folder, 600, max_attempts=1, interrupted=True)
    resumed = load_ssw_checkpoint(folder / 'checkpoint.pkl')
    assert resumed.next_index == saved.next_index + 1
    assert state['search'] == resumed.evaluation_requests + 64
    assert state['unconfirmed_search_reservations'] == 64
    assert state['reserved_seconds'] == 1800


def test_terminal_budget_preserves_previous_safe_checkpoint(tmp_path):
    probe = tmp_path / 'probe'; prepare(probe)
    cost = r.run_segment(probe, 600, max_attempts=1)['search']
    folder = tmp_path / 'arm'; plan = prepare(folder)
    plan['search_cap'] = cost + 1
    r.atomic_json(folder / 'plan.json', plan)
    r.run_segment(folder, 600, max_attempts=1)
    state = r.run_segment(folder, 600, max_attempts=1)
    assert state['status'] == 'search_exhausted'
    assert state['search'] == cost + 1
    assert load_ssw_checkpoint(folder / 'checkpoint.pkl').next_index == 1
    assert load_ssw_checkpoint(folder / 'terminal.pkl').status != 'completed'
    with pytest.raises(ValueError, match='terminal'):
        r.run_segment(folder, 600, max_attempts=1)


def test_finalize_fresh_budget_is_persistent_and_not_repeated(tmp_path):
    import shutil
    folder = tmp_path / 'arm'; plan = prepare(folder)
    plan['reference_energy_eV'] = 100.
    r.atomic_json(folder / 'plan.json', plan)
    validator = Path(__file__).parents[2] / 'research/ga_ssw/evidence/c60-recovered-rotation-ls-prospective-20260922/validator.py'
    shutil.copy2(validator, folder / 'validator.py')
    r.run_segment(folder, 600, max_attempts=1)
    state = r.Budget(folder, plan); state.state['status'] = 'completed'; state.save()
    r.finalize(folder)
    first = json.loads((folder / 'summary.json').read_text())
    r.finalize(folder)
    second = json.loads((folder / 'summary.json').read_text())
    assert first['fresh'] == second['fresh'] == 2
    assert first['fresh_checks'] == second['fresh_checks']
    assert all(v['numerical_qualified'] for v in second['fresh_checks'].values())
    assert not any(v['joint_target'] for v in second['fresh_checks'].values())


def test_submission_is_once_and_stages_cannot_overlap(tmp_path, monkeypatch):
    import runpy
    import shutil
    import subprocess
    import sys
    from types import SimpleNamespace
    root = Path(__file__).parents[2]
    source = root / 'research/ga_ssw/evidence/c60-long-budget-20260924'
    shutil.copy2(source / 'submit_stage.py', tmp_path)
    shutil.copy2(root / 'research/ga_ssw/c60_long_budget.py', tmp_path / 'production_runner.py')
    r.atomic_json(tmp_path / 'preflight-qualification.json', {'status': 'passed'})
    r.atomic_json(tmp_path / 'submission-ledger.json', {'submissions': []})
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=str(100 + len(calls)), stderr='')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['submit_stage.py', '--stage', '1'])
    runpy.run_path(str(tmp_path / 'submit_stage.py'), run_name='__main__')
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match='already attempted'):
        runpy.run_path(str(tmp_path / 'submit_stage.py'), run_name='__main__')
    assert len(calls) == 2
    monkeypatch.setattr(sys, 'argv', ['submit_stage.py', '--stage', '2'])
    runpy.run_path(str(tmp_path / 'submit_stage.py'), run_name='__main__')
    assert '--dependency=afterany:101' in calls[2]
    assert '--dependency=afterany:103' in calls[3]
