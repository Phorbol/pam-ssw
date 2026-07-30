import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260730-direction-mode-continuation"
)
RUNNER_PATH = RUN_ROOT / "run_residual_observation.py"
ANALYZER_PATH = RUN_ROOT / "analyze_residual_observation.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_residual_observation_protocol_is_transport_only_and_thirty_actions():
    runner = _load(RUNNER_PATH, "_transport_residual_runner")

    assert runner.SEEDS == tuple(range(42, 52))
    assert runner.ARM == "transported_direction"
    assert runner.case_matrix("c60") == [
        {
            "system": "c60",
            "state_id": state_id,
            "seed": seed,
            "arm": "transported_direction",
        }
        for state_id in ("intermediate_accepted", "plateau_accepted")
        for seed in range(42, 52)
    ]
    assert runner.case_matrix("pdo") == [
        {
            "system": "pdo",
            "state_id": "raw_bootstrap",
            "seed": seed,
            "arm": "transported_direction",
        }
        for seed in range(42, 52)
    ]


def _case(system, state_id, seed, *, reverse_signal=False):
    level = float(seed - 41)
    residual = level / 10.0
    if reverse_signal and system == "pdo":
        downstream = int(500.0 - 100.0 * residual)
    else:
        downstream = int(100.0 + 100.0 * residual)
    proposal = downstream // 2
    landing = downstream - proposal
    purposes = {
        "bootstrap_true_quench": 0,
        "starter_true_quench": 0,
        "direction_oracle": 4,
        "escape_true_pes_check": 2,
        "biased_proposal_relax": proposal,
        "landing_true_quench": landing,
        "post_relax_validation": 1,
        "unattributed": 0,
    }
    return {
        "system": system,
        "state_id": state_id,
        "seed": seed,
        "arm": "transported_direction",
        "status": "completed",
        "certificate": True,
        "force_evaluations": sum(purposes.values()),
        "purpose_counts": purposes,
        "quench_iterations": landing,
        "landing_delta_eV": -0.01 * level,
        "relaxation_diagnostics": {
            "proposal_relax_count": 2,
            "proposal_relax_mean_iterations": proposal / 2.0,
            "proposal_relax_outcome_energy_exploded": int(
                downstream > 170
            ),
            "proposal_relax_unconverged": 0,
            "true_quench_count": 1,
            "true_quench_mean_iterations": float(landing),
            "true_quench_unconverged": 0,
            "quench_fallback_attempts": 0,
            "quench_fallback_converged": 0,
        },
        "direction_trace": [
            {
                "step": 0,
                "selected_kind": "block_ritz",
                "shared_initial_direction": True,
                "oracle_selection_force_evaluations_delta": 0,
            },
            {
                "step": 1,
                "selected_kind": "transported",
                "direction_hvp_count": 1,
                "oracle_selection_force_evaluations_delta": 2,
                "transported_relative_residual": 0.5 * residual,
                "transported_true_relative_residual": 0.4 * residual,
            },
            {
                "step": 2,
                "selected_kind": "transported",
                "direction_hvp_count": 1,
                "oracle_selection_force_evaluations_delta": 2,
                "transported_relative_residual": residual,
                "transported_true_relative_residual": 0.8 * residual,
            },
        ],
    }


def _cohort(*, reverse_signal=False):
    rows = []
    for state_id in ("intermediate_accepted", "plateau_accepted"):
        for seed in range(42, 52):
            rows.append(
                _case(
                    "c60",
                    state_id,
                    seed,
                    reverse_signal=reverse_signal,
                )
            )
    for seed in range(42, 52):
        rows.append(
            _case(
                "pdo",
                "raw_bootstrap",
                seed,
                reverse_signal=reverse_signal,
            )
        )
    return rows


def test_residual_analysis_survives_only_for_cross_system_cost_ordering():
    analyzer = _load(
        ANALYZER_PATH,
        "_transport_residual_analyzer_support",
    )

    evidence = analyzer.analyze_cases(_cohort())

    assert evidence["decision"] == "residual_signal_consistent"
    assert evidence["systems"]["c60"][
        "spearman_max_residual_vs_downstream_fe"
    ] > 0.0
    assert evidence["systems"]["pdo"][
        "spearman_max_residual_vs_downstream_fe"
    ] > 0.0
    assert evidence["systems"]["c60"][
        "median_residual_change"
    ] > 0.0
    assert evidence["zero_extra_direction_fe"] is True


def test_residual_analysis_rejects_opposite_system_ordering():
    analyzer = _load(
        ANALYZER_PATH,
        "_transport_residual_analyzer_reject",
    )

    evidence = analyzer.analyze_cases(
        _cohort(reverse_signal=True)
    )

    assert evidence["decision"] == "residual_signal_inconsistent"
    assert evidence["systems"]["pdo"][
        "spearman_max_residual_vs_downstream_fe"
    ] < 0.0
