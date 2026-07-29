from __future__ import annotations

import json
from pathlib import Path

from benchmarks.bias_relaxation_compare import BACKENDS, run_comparison


def test_run_comparison_emits_raw_per_case_schema_without_rankings(tmp_path: Path):
    output = tmp_path / "comparison.json"

    payload = run_comparison(output)

    assert payload["schema_version"] == 1
    assert payload["backends"] == list(BACKENDS)
    assert len(payload["cases"]) == 2
    assert output.exists()
    assert json.loads(output.read_text()) == payload
    for case in payload["cases"]:
        assert set(case) == {
            "case_id",
            "initial_numbers",
            "initial_positions",
            "cell",
            "pbc",
            "fixed_mask",
            "hessian_diagonal",
            "biases",
            "fmax",
            "maxiter",
            "runs",
        }
        assert len(case["hessian_diagonal"]) == 3 * len(case["initial_numbers"])
        assert case["biases"]
        assert [run["backend"] for run in case["runs"]] == list(BACKENDS)
        for run in case["runs"]:
            assert set(run) == {
                "backend",
                "available",
                "objective_calls",
                "telemetry_evaluator_calls",
                "backend_evaluations",
                "converged",
                "termination_reason",
                "iterations",
                "final_energy",
                "final_max_force",
                "accepted_steps",
                "rejected_steps",
                "accepted_secants",
                "rejected_secants",
                "line_search_evaluations",
                "mic_branch_resets",
                "bias_secant_curvature_sum",
                "final_positions",
            }
            if run["backend"] in {"ase-fire", "ase-fire2"}:
                assert run["accepted_steps"] is None
                assert run["accepted_secants"] is None
                assert run["line_search_evaluations"] is None

    serialized = json.dumps(payload).lower()
    assert "rank" not in serialized
    assert "winner" not in serialized
    assert "score" not in serialized


def test_deterministic_custom_runs_close_objective_call_ledger_and_converge(tmp_path: Path):
    payload = run_comparison(tmp_path / "comparison.json")

    for case in payload["cases"]:
        for run in case["runs"]:
            if run["backend"] not in {"safe-lbfgs-total", "bias-separated-lbfgs"}:
                continue
            assert run["available"]
            assert run["objective_calls"] == run["telemetry_evaluator_calls"]
            assert run["objective_calls"] == run["backend_evaluations"]
            assert run["converged"], (case["case_id"], run)
            assert run["termination_reason"] == "converged"
            assert run["final_max_force"] <= case["fmax"]
