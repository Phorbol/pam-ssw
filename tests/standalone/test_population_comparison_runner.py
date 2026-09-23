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
