import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from research.ga_ssw.run_population_comparison import (
    CountedSurface, SearchStopped, allocate_remaining, initial_quench_options,
    run_walk_allocations,
)


class MockSurface:
    def __init__(self, fail=False):
        self.requests = 0
        self.fail = fail

    def evaluate(self, atoms):
        self.requests += 1
        if self.fail:
            raise ValueError("mock calculator failure")
        return 1.25, np.zeros_like(atoms.positions)


def test_allocate_remaining_is_even_and_fixed_order():
    assert allocate_remaining(10, 3) == [4, 3, 3]
    assert allocate_remaining(2, 4) == [1, 1, 0, 0]
    assert allocate_remaining(0, 2) == [0, 0]
    assert allocate_remaining(5, 0) == []


def test_counted_surface_counts_failure_and_refusal_does_not_spend(tmp_path):
    ledger = tmp_path / "evaluations.jsonl"
    wrapped = MockSurface(fail=True)
    counted = CountedSurface(wrapped, cap=1, wall_seconds=60,
                              started=time.monotonic(), ledger_path=ledger)
    # Use a started time of zero and a long wall bound so the request cap is
    # the only stopping condition in this deterministic mock test.
    with pytest.raises(ValueError, match="mock calculator failure"):
        counted.evaluate(Atoms("H", positions=[[0, 0, 0]]))
    assert counted.requests == wrapped.requests == 1
    with pytest.raises(SearchStopped, match="search_cap"):
        counted.evaluate(Atoms("H", positions=[[0, 0, 0]]))
    assert counted.requests == wrapped.requests == 1
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["failed_request", "refused"]
    assert rows[-1]["request_index"] == 2


def test_counted_surface_success_logs_phase_and_request(tmp_path):
    ledger = tmp_path / "evaluations.jsonl"
    wrapped = MockSurface()
    counted = CountedSurface(wrapped, cap=2, wall_seconds=60,
                              started=time.monotonic(), ledger_path=ledger)
    counted.set_phase("ssw:0")
    energy, forces = counted.evaluate(Atoms("H", positions=[[0, 0, 0]]))
    assert energy == 1.25
    assert forces.shape == (1, 3)
    row = json.loads(ledger.read_text().splitlines()[0])
    assert row["kind"] == "request"
    assert row["request_index"] == 1
    assert row["phase"] == "ssw:0"


def test_walk_allocations_carry_unused_and_keep_last_valid_result():
    starts = [(i, object()) for i in range(3)]
    valid_last_minima = []

    def mock_walk(index, quench_result, allowance):
        result = {"minima": [f"valid-minimum-{index}"]}
        valid_last_minima.extend(result["minima"])
        return {"status": "budget_exhausted", "used": allowance,
                "request_start": index * 10 + 1, "request_end": index * 10 + allowance,
                "result": result}

    rows, left = run_walk_allocations(starts, 10, mock_walk)
    assert [row["allocation"] for row in rows] == [4, 3, 3]
    assert [row["used"] for row in rows] == [4, 3, 3]
    assert left == 0
    assert valid_last_minima == ["valid-minimum-0", "valid-minimum-1", "valid-minimum-2"]
    assert rows[-1]["result"]["minima"] == ["valid-minimum-2"]

    usage = iter((2, 4, 4))
    carried, left = run_walk_allocations(starts, 10, lambda index, q, allocation: {
        "status": "completed", "used": next(usage), "result": {"last": index},
        "request_start": 1, "request_end": allocation,
    })
    assert [row["allocation"] for row in carried] == [4, 4, 4]
    assert [row["unused_carried"] for row in carried] == [2, 0, 0]
    assert left == 0


def test_initial_quench_optimizer_matches_ga_selection():
    safe = initial_quench_options(SimpleNamespace(
        quench_optimizer="safe-lbfgs-total", lbfgs_memory=500))
    assert safe == {"optimizer": "safe-lbfgs-total", "lbfgs_memory": 500}
    default = initial_quench_options(SimpleNamespace(
        quench_optimizer="ase-lbfgs", lbfgs_memory=None))
    from ase.optimize import BFGS
    assert default == {"optimizer": BFGS, "lbfgs_memory": None}


def test_ga_wall_guard_stops_at_existing_boundary_without_pes(tmp_path):
    """A wall stop must not launch proposal/fine phases or masquerade as completion."""
    from ase.io import write
    from research.ga_ssw.run_population_comparison import run
    inputs = []
    for index in range(3):
        path = tmp_path / f'input{index}.extxyz'
        write(path, Atoms('Cu13', positions=[[i * (2.5 + .1 * index), 0, 0] for i in range(13)]))
        inputs.append(str(path))
    plan = dict(inputs=inputs, backend={'kind': 'emt'}, seed=3, search_cap=100,
        wall_seconds=1e-12, outer_steps=10,
        ssw=dict(width=.2, rotation_bias=1., max_gaussians=1, temperature_K=300.,
                 fmax=.03, relax_steps=20, fd_step=.001, rotation_hvp=4,
                 rotation_tol=.02, cluster_frame='direction_only'),
        ga=dict(quick_steps=1, generations=1, generation_steps=1, fine_steps=1,
                ga_candidates=8, regions=1, fine_regions=1, quench_fmax=.03,
                quench_steps=20, proposal_max_batches=1, proposal_max_cut_attempts=20,
                proposal_max_pair_attempts=20, partition_max_draws=20,
                projection_tolerance=.001, energy_window=10., proposal_type=0),
        descriptor=dict(bond_lengths=[[29,29,2.26]],neighbor_range=2.,weights=[.3,.2,.2,.1,.1,.1]))
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan))
    out = tmp_path/'run'
    assert run(plan_path, 'ga', out) == 0
    summary = json.loads((out/'summary.json').read_text())
    assert summary['status'] == 'wall_censored'
    assert summary['algorithm_status'] == 'checkpoint_boundary'
    assert summary['search_requests'] == 0
    assert not any(s['phase'] in ('offspring_quench','fine','ga_proposal') for s in summary['stages'])
    assert summary['fresh_checks'] == []


def test_invalid_native_mc_option_fails_before_backend_creation(tmp_path, monkeypatch):
    from ase.io import write
    import research.ga_ssw.run_population_comparison as runner

    inputs = []
    for index in range(3):
        path = tmp_path / f'input{index}.extxyz'
        write(path, Atoms('Cu13', positions=[[i * (2.5 + .1 * index), 0, 0] for i in range(13)]))
        inputs.append(str(path))
    plan = dict(inputs=inputs, backend={'kind': 'emt'}, seed=3, search_cap=100,
        wall_seconds=60, outer_steps=10,
        ssw=dict(width=.2, rotation_bias=1., max_gaussians=1, temperature_K=300.,
                 fmax=.03, relax_steps=20, fd_step=.001, rotation_hvp=4,
                 rotation_tol=.02, cluster_frame='direction_only'),
        ga=dict(quick_steps=1, generations=1, generation_steps=1, fine_steps=1,
                ga_candidates=8, regions=1, fine_regions=1, quench_fmax=.03,
                quench_steps=20, proposal_max_batches=1, proposal_max_cut_attempts=20,
                proposal_max_pair_attempts=20, partition_max_draws=20,
                projection_tolerance=.001, energy_window=10., proposal_type=0),
        native_mc={'energy_tol_eV': .001, 'maxtrap': 10},
        descriptor=dict(bond_lengths=[[29,29,2.26]], neighbor_range=2.,
                        weights=[.3,.2,.2,.1,.1,.1]))
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    backend_calls = []
    monkeypatch.setattr(runner, 'build_backend', lambda settings: backend_calls.append(settings))

    with pytest.raises(TypeError, match='energy_tol_eV'):
        runner.run(plan_path, 'ssw', tmp_path / 'run')

    assert backend_calls == []
    failure = json.loads((tmp_path / 'run' / 'failure.json').read_text())
    assert failure['search_requests'] == 0
    assert 'energy_tol_eV' in failure['error']
    assert not (tmp_path / 'run' / 'evaluations.jsonl').exists()
