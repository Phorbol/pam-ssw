import importlib.util
from pathlib import Path


RUNNER = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("vc_atomic_runner_contract", RUNNER)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_treatment_configs_differ_only_in_atomic_period():
    config = RUN.treatment_configurations()
    cell = config["cell_only"]
    joint = config["cell_plus_atomic"]
    assert cell["block"].atomic_period == 21
    assert joint["block"].atomic_period == 2
    assert cell["atomic"].rotation_exit_policy == "force_or_budget"
    assert joint["atomic"].rotation_exit_policy == "force_or_budget"
    cell_values = vars(cell["block"])
    joint_values = vars(joint["block"])
    assert {key for key in cell_values if cell_values[key] != joint_values[key]} == {"atomic_period"}
    cell_atomic = vars(cell["atomic"])
    joint_atomic = vars(joint["atomic"])
    assert cell_atomic == joint_atomic
    assert RUN._config_row(config)["cell_plus_atomic"]["block"]["atomic_period"] == 2
    assert RUN._config_row({"cell_only": cell})["cell_only"]["block"]["atomic_period"] == 21


def test_budget_censor_does_not_hide_paid_failure():
    records = [
        {"index": 0, "requests": 4, "status": "completed"},
        {"index": 1, "requests": 2, "status": "cell_rotation_failed"},
        {"index": 2, "requests": 0, "status": "cell_rotation_failed"},
    ]
    ledger = [
        {"event": "failure", "request": 6, "error": "RuntimeError('backend')"},
        {"event": "denial", "request": 6, "reason": "per_case_request_cap"},
    ]
    outcomes = RUN.classify_outer_records(records, ledger)
    assert [row["outcome"] for row in outcomes] == ["completed", "paid_failure", "not_run_after_paid_failure"]


def test_budget_denial_and_zero_cost_tail_are_distinct():
    records = [
        {"index": 0, "requests": 4, "status": "completed"},
        {"index": 1, "requests": 1, "status": "cell_rotation_failed"},
        {"index": 2, "requests": 0, "status": "cell_rotation_failed"},
    ]
    ledger = [{"event": "denial", "request": 5, "reason": "per_case_request_cap"}]
    outcomes = RUN.classify_outer_records(records, ledger)
    assert [row["outcome"] for row in outcomes] == ["completed", "budget_censored", "not_started_no_budget"]


def test_valid_landing_at_cap_precedes_next_outer_denial():
    records = [
        {"index": 0, "requests": 5, "status": "valid_landing"},
        {"index": 1, "requests": 0, "status": "evaluation_failed"},
    ]
    ledger = [{"event": "denial", "request": 5, "reason": "per_case_request_cap"}]
    outcomes = RUN.classify_outer_records(records, ledger)
    assert [row["outcome"] for row in outcomes] == ["valid_landing", "not_started_no_budget"]


def test_stage_diagnostic_identifies_blocked_atomic_vs_attempted_climb():
    blocked = RUN.atomic_path_diagnostic({
        "atomic_scheduled": True,
        "cell_cycles": [{"status": "completed"}, {"status": "partial_atom_relax_failed"}],
        "atomic": None,
        "landing": None,
    })
    attempted = RUN.atomic_path_diagnostic({
        "atomic_scheduled": True,
        "cell_cycles": [{"status": "completed"}],
        "atomic": type("Atomic", (), {"climb": [
            {"rotation_budget_released": True, "true_energy": -1.0,
             "quench_status": "converged"}
        ]})(),
        "landing": type("Landing", (), {"converged": False})(),
    })
    quench_failure = RUN.atomic_path_diagnostic({
        "atomic_scheduled": True,
        "cell_cycles": [{"status": "completed"}],
        "atomic": type("Atomic", (), {"status": "biased_quench_failed", "climb": [
            {"rotation_budget_released": True, "status": "biased_quench_failed"}
        ]})(),
        "landing": None,
    })
    assert blocked == "atomic_not_reached_after_cell_failure"
    assert attempted == "completed_gaussians_then_landing_failed"
    assert quench_failure == "biased_quench_failed_after_0_completed_gaussians"


def test_pair_trace_requires_initial_first_outer_and_pre_atomic_agreement():
    row = {"outer_trace": [
        {"index": 0, "cell_cycles": [{"direction": [1.0, 0.0]}]},
        {"index": 1, "cell_cycles": [{"direction": [0.0, 1.0]}]},
    ], "initial_energy_eV": -1.25,
        "initial_structure_signature": {"positions_A": [[0.0, 1.0]], "cell_A": [[2.0]], "numbers": [8]}}
    assert RUN.paired_trace_check(row, row)["pre_atomic_cell_stage"]["matches_within_trace_tolerance"]
    near = {"outer_trace": [
        {"index": 0, "cell_cycles": [{"direction": [1.0, 0.0]}]},
        {"index": 1, "cell_cycles": [{"direction": [0.0, 1.0 + 1e-10]}]},
    ], "initial_energy_eV": -1.25,
        "initial_structure_signature": row["initial_structure_signature"]}
    assert RUN.paired_trace_check(row, near)["pre_atomic_cell_stage"]["matches_within_trace_tolerance"]
    changed = {"outer_trace": [
        {"index": 0, "cell_cycles": [{"direction": [1.0, 0.0]}]},
        {"index": 1, "cell_cycles": [{"direction": [0.0, 1.1]}]},
    ], "initial_energy_eV": -1.25}
    changed["initial_structure_signature"] = row["initial_structure_signature"]
    assert not RUN.paired_trace_check(row, changed)["pre_atomic_cell_stage"]["matches_within_trace_tolerance"]
