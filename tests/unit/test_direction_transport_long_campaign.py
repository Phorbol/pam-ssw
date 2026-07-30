import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260730-direction-mode-continuation"
)
RUNNER_PATH = RUN_ROOT / "run_long_campaign.py"
ANALYZER_PATH = RUN_ROOT / "analyze_long_campaign.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_long_campaign_protocol_is_the_four_case_seed42_screen():
    runner = _load(RUNNER_PATH, "_direction_transport_long_runner")

    assert runner.SYSTEMS == ("c60", "pdo")
    assert runner.SEEDS == (42,)
    assert runner.ARMS == {
        "fixed_intent_ritz": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
        "transported_direction": {
            "direction_selection_mode": "transported_direction",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
    }
    assert runner.case_matrix() == [
        {"system": system, "seed": 42, "arm": arm}
        for system in ("c60", "pdo")
        for arm in ("fixed_intent_ritz", "transported_direction")
    ]


def _summary(system, arm, *, force_evaluations, best, auc):
    direction = 30 if arm == "fixed_intent_ritz" else 5
    purposes = {
        "bootstrap_true_quench": 10,
        "starter_true_quench": 0,
        "direction_oracle": direction,
        "escape_true_pes_check": 5,
        "biased_proposal_relax": 20,
        "landing_true_quench": force_evaluations - 36 - direction,
        "post_relax_validation": 1,
        "unattributed": 0,
    }
    return {
        "system": system,
        "seed": 42,
        "arm": arm,
        "completed_trials": 200,
        "recorded_walks": 200,
        "initial_energy_eV": -100.0,
        "best_energy_eV": best,
        "best_energy_drop_eV": -100.0 - best,
        "mean_best_energy_improvement_eV": auc,
        "archive_entries": 100,
        "duplicate_rate": 0.5,
        "force_evaluations": force_evaluations,
        "purpose_counts": purposes,
        "wall_time_s": float(force_evaluations),
        "optimizer_telemetry": {
            "true_quench_unconverged": 0,
            "quench_fallback_attempts": 0,
            "quench_fallback_converged": 0,
        },
        "stats": {
            "n_trials": 200,
            "fragment_rejections": 0,
        },
    }


def test_long_campaign_analysis_uses_vector_pareto_gates():
    analyzer = _load(ANALYZER_PATH, "_direction_transport_long_analyzer")
    rows = []
    for system in ("c60", "pdo"):
        rows.extend(
            [
                _summary(
                    system,
                    "fixed_intent_ritz",
                    force_evaluations=100,
                    best=-105.0,
                    auc=3.0,
                ),
                _summary(
                    system,
                    "transported_direction",
                    force_evaluations=80,
                    best=-106.0,
                    auc=4.0,
                ),
            ]
        )

    evidence = analyzer.analyze(rows)

    assert evidence["decision"] == "seed42_screen_survives"
    assert evidence["systems"]["c60"]["decision"] == "pareto_dominates"
    assert evidence["systems"]["pdo"]["decision"] == "pareto_dominates"
    assert evidence["systems"]["c60"]["force_evaluations_saved"] == 20


def test_lower_cost_with_worse_search_is_reported_as_tradeoff():
    analyzer = _load(ANALYZER_PATH, "_direction_transport_long_analyzer_tradeoff")
    rows = [
        _summary(
            "c60",
            "fixed_intent_ritz",
            force_evaluations=100,
            best=-105.0,
            auc=3.0,
        ),
        _summary(
            "c60",
            "transported_direction",
            force_evaluations=80,
            best=-104.0,
            auc=2.0,
        ),
    ]

    result = analyzer.compare_system(rows)

    assert result["decision"] == "cost_search_tradeoff"
