"""Fail-closed contract tests for the sequential strict-quench rescue analysis."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "20260728-true-quench-sequential-rescue"
ANALYZER_PATH = RUN_ROOT / "analyze.py"
CORPUS_PATH = RUN_ROOT / "corpus.json"
EXECUTION_COMMIT = "42054840dd58701e89005ef06a6f0eb082cd2603"
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
PURPOSES = (
    "bootstrap_true_quench",
    "starter_true_quench",
    "direction_oracle",
    "escape_true_pes_check",
    "biased_proposal_relax",
    "landing_true_quench",
    "post_relax_validation",
    "unattributed",
)
ARM_RESULTS = {
    "ase-lbfgs-restart": {
        "optimizer": "ase-lbfgs",
        "history": None,
        "force_evaluations": (72, 11, 402, 235),
        "success": (True, True, False, True),
        "wall_time_s": (12.3, 0.2, 8.8, 5.1),
        "energy_change_eV": (-2.75, -0.002, 0.015, -0.93),
    },
    "safe-lbfgs-total": {
        "optimizer": "safe-lbfgs-total",
        "history": 10,
        "force_evaluations": (102, 44, 45, 83),
        "success": (False, False, False, True),
        "wall_time_s": (1.7, 0.7, 1.0, 1.8),
        "energy_change_eV": (-2.75, -0.002, -0.001, -0.93),
    },
    "ase-fire": {
        "optimizer": "ase-fire",
        "history": None,
        "force_evaluations": (268, 23, 119, 84),
        "success": (True, True, True, True),
        "wall_time_s": (4.6, 0.4, 2.6, 1.8),
        "energy_change_eV": (-2.75, -0.002, -0.28, -0.07),
    },
}


def _load_analyzer(name: str):
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, monkeypatch) -> tuple[object, Path, Path, list[dict], dict]:
    analyzer = _load_analyzer(f"sequential_rescue_analysis_{tmp_path.name}")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for arm_id, arm in ARM_RESULTS.items():
        for index, entry in enumerate(corpus["entries"]):
            passed = arm["success"][index]
            final_force = 0.009 if passed else 0.02
            termination = "converged" if passed else (
                "maxiter" if arm_id == "ase-lbfgs-restart" else "line_search_failed"
            )
            force_evaluations = arm["force_evaluations"][index]
            initial_energy = float(entry["fallback_start"]["energy_eV"])
            energy_change = arm["energy_change_eV"][index]
            purpose = {name: 0 for name in PURPOSES}
            purpose["landing_true_quench"] = force_evaluations
            rows.append(
                {
                    "system": entry["system"],
                    "task_index": entry["task_index"],
                    "trial_index": entry["trial_index"],
                    "proposal_index": entry["proposal_index"],
                    "source_trajectory_path": entry["source_trajectory_path"],
                    "source_trajectory_sha256": entry["source_trajectory_sha256"],
                    "source_frame_index": entry["source_frame_index"],
                    "arm_id": arm_id,
                    "optimizer": arm["optimizer"],
                    "safe_history_limit": arm["history"],
                    "fmax_eV_per_A": 0.01,
                    "maxiter": 400,
                    "coordinate_trust_radius_A": None,
                    "objective": "true_mace_pes_no_bias_no_softening",
                    "primary": deepcopy(entry["primary"]),
                    "fallback": {
                        "initial": {
                            "energy_eV": initial_energy,
                            "max_active_force_eV_per_A": float(
                                entry["fallback_start"]["max_active_force_eV_per_A"]
                            ),
                            "positions_sha256": entry["fallback_start"][
                                "positions_sha256"
                            ],
                            "state": deepcopy(entry["fallback_start"]["state"]),
                        },
                        "final": {
                            "energy_eV": initial_energy + energy_change,
                            "max_active_force_eV_per_A": final_force,
                            "positions_sha256": entry["fallback_start"][
                                "positions_sha256"
                            ],
                            "state": deepcopy(entry["fallback_start"]["state"]),
                        },
                        "energy_change_eV": energy_change,
                        "certificate_passed": passed,
                        "n_iter": force_evaluations - 1,
                        "termination_reason": termination,
                        "outcome_class": (
                            "converged_productive" if passed else "useful_progress"
                        ),
                        "telemetry": {
                            "backend": arm["optimizer"],
                            "converged": passed,
                            "evaluator_calls": force_evaluations,
                            "gradient_measure": "raw_active_max_force",
                            "termination_reason": termination,
                        },
                        "evaluator_calls": force_evaluations,
                        "force_evaluations": force_evaluations,
                        "purpose_count_delta": purpose,
                        "wall_time_s": arm["wall_time_s"][index],
                    },
                    "cost": {
                        "primary_force_evaluations": entry["primary"][
                            "force_evaluations"
                        ],
                        "offline_fallback_force_evaluations": force_evaluations,
                        "offline_combined_replay_force_evaluations": (
                            entry["primary"]["force_evaluations"] + force_evaluations
                        ),
                        "offline_repeats_primary_terminal_evaluation": True,
                        "continuous_implementation_avoidable_force_evaluations": 1,
                    },
                }
            )
    fallback_total = sum(
        row["fallback"]["force_evaluations"] for row in rows
    )
    fallback_counts = {name: 0 for name in PURPOSES}
    fallback_counts["landing_true_quench"] = fallback_total
    primary = corpus["selection"]["primary_cohort"]
    primary_total = primary["overall"]["total_force_evaluations"]
    summary = {
        "schema_version": 1,
        "execution_commit": EXECUTION_COMMIT,
        "corpus": {
            "path": "runs/20260728-true-quench-sequential-rescue/corpus.json",
            "sha256": sha256(CORPUS_PATH.read_bytes()).hexdigest(),
            "entry_count": 4,
        },
        "source_replay": deepcopy(corpus["source"]),
        "primary_cohort": deepcopy(primary),
        "model": deepcopy(corpus["source"]["model"]),
        "runtime_versions": deepcopy(corpus["source"]["runtime_versions"]),
        "cuda": deepcopy(corpus["source"]["cuda"]),
        "calculator": deepcopy(corpus["source"]["calculator"]),
        "safe_lbfgs_default_history_limit": 10,
        "strict_protocol": {
            "fmax_eV_per_A": 0.01,
            "maxiter": 400,
            "coordinate_trust_radius_A": None,
            "objective": "true_mace_pes_no_bias_no_softening",
        },
        "arms": [
            {
                "arm_id": arm_id,
                "optimizer": arm["optimizer"],
                "safe_history_limit": arm["history"],
            }
            for arm_id, arm in ARM_RESULTS.items()
        ],
        "trigger_count": 4,
        "row_count": 12,
        "calculator_instances": 6,
        "fallback_evaluation_counts": fallback_counts,
        "pipeline_cost_by_fallback_arm": {
            arm_id: {
                "trigger_count": 4,
                "fallback_certificate_success_count": sum(arm["success"]),
                "offline_fallback_force_evaluations": sum(
                    arm["force_evaluations"]
                ),
                "offline_pipeline_total_force_evaluations": (
                    primary_total + sum(arm["force_evaluations"])
                ),
                "continuous_pipeline_projected_force_evaluations": (
                    primary_total + sum(arm["force_evaluations"]) - 4
                ),
            }
            for arm_id, arm in ARM_RESULTS.items()
        },
        "rows_file": "rows.json",
        "cost_interpretation": "fixed",
        "claim_boundary": "fixed",
    }
    rows_path = tmp_path / "rows.json"
    summary_path = tmp_path / "summary.json"
    _write_json(rows_path, rows)
    _write_json(summary_path, summary)
    monkeypatch.setattr(
        analyzer, "EXPECTED_ROWS_SHA256", sha256(rows_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        analyzer,
        "EXPECTED_SUMMARY_SHA256",
        sha256(summary_path.read_bytes()).hexdigest(),
    )
    return analyzer, summary_path, rows_path, rows, summary


def test_analyzer_reports_fixed_fallback_and_pipeline_costs(tmp_path, monkeypatch):
    analyzer, summary_path, rows_path, _, _ = _fixture(tmp_path, monkeypatch)

    evidence = analyzer.analyze(
        summary_path=summary_path,
        rows_path=rows_path,
        corpus_path=CORPUS_PATH,
    )

    assert evidence["validation"]["status"] == "passed"
    assert evidence["validation"]["row_count"] == 12
    assert evidence["validation"]["primary_cohort"]["overall"] == {
        "task_count": 32,
        "certificate_success_count": 28,
        "trigger_count": 4,
        "total_force_evaluations": 5096,
        "total_wall_time_s": pytest.approx(98.84687816997757),
    }
    expected = {
        "ase-lbfgs-restart": (3, 720, 5816, 5812),
        "safe-lbfgs-total": (1, 274, 5370, 5366),
        "ase-fire": (4, 494, 5590, 5586),
    }
    for arm_id, (success, fallback_fe, offline, continuous) in expected.items():
        arm = evidence["arms"][arm_id]
        assert arm["fallback"]["task_count"] == 4
        assert arm["fallback"]["certificate_success_count"] == success
        assert arm["fallback"]["total_force_evaluations"] == fallback_fe
        assert arm["fallback"]["total_wall_time_s"] > 0
        assert sum(arm["fallback"]["termination_reason_counts"].values()) == 4
        assert arm["fallback"]["energy_change_eV"]["count"] == 4
        assert arm["fallback"]["final_max_force_eV_per_A"]["count"] == 4
        assert arm["pipeline"]["certificate_success_count"] == 28 + success
        assert arm["pipeline"]["offline_total_force_evaluations"] == offline
        assert (
            arm["pipeline"]["continuous_projected_total_force_evaluations"]
            == continuous
        )
        assert (
            arm["pipeline"]["continuous_projected_increment_force_evaluations"]
            == continuous - 5096
        )
        assert arm["pipeline"][
            "continuous_projected_increment_fraction_of_primary"
        ] == pytest.approx((continuous - 5096) / 5096)
    assert evidence["arms"]["ase-fire"]["pipeline"][
        "continuous_projected_increment_fraction_of_primary"
    ] == pytest.approx(490 / 5096)
    assert evidence["selection_statement"] == {
        "only_full_coverage_fallback_on_fixed_corpus": "ase-fire",
        "production_default_selected": False,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda summary, rows: summary.__setitem__(
                "execution_commit", "0" * 40
            ),
            "execution commit",
        ),
        (
            lambda summary, rows: summary["model"].__setitem__(
                "sha256", "0" * 64
            ),
            "model",
        ),
        (
            lambda summary, rows: summary["model"].__setitem__(
                "path", "/different-model"
            ),
            "model",
        ),
        (
            lambda summary, rows: summary["cuda"].__setitem__(
                "device_name", "different GPU"
            ),
            "CUDA",
        ),
        (
            lambda summary, rows: summary["runtime_versions"].__setitem__(
                "torch", "different"
            ),
            "runtime",
        ),
        (
            lambda summary, rows: rows.pop(),
            "row count",
        ),
        (
            lambda summary, rows: rows[0].__setitem__(
                "source_frame_index", 99
            ),
            "raw fallback start identity",
        ),
        (
            lambda summary, rows: rows[0]["fallback"]["final"].__setitem__(
                "energy_eV", "not-finite"
            ),
            "finite",
        ),
        (
            lambda summary, rows: rows[0]["fallback"].__setitem__(
                "certificate_passed", False
            ),
            "certificate",
        ),
        (
            lambda summary, rows: rows[0]["fallback"]["telemetry"].__setitem__(
                "converged", False
            ),
            "telemetry",
        ),
        (
            lambda summary, rows: rows[0]["fallback"][
                "purpose_count_delta"
            ].__setitem__("unattributed", 1),
            "evaluation ledger",
        ),
        (
            lambda summary, rows: summary[
                "pipeline_cost_by_fallback_arm"
            ]["ase-fire"].__setitem__(
                "continuous_pipeline_projected_force_evaluations", 1
            ),
            "summary per-arm",
        ),
        (
            lambda summary, rows: summary["primary_cohort"]["overall"].__setitem__(
                "total_force_evaluations", 1
            ),
            "primary cohort",
        ),
    ),
)
def test_analyzer_fails_closed(
    tmp_path, monkeypatch, mutation, match: str
):
    analyzer, summary_path, rows_path, rows, summary = _fixture(
        tmp_path, monkeypatch
    )
    mutation(summary, rows)
    _write_json(rows_path, rows)
    _write_json(summary_path, summary)
    monkeypatch.setattr(
        analyzer, "EXPECTED_ROWS_SHA256", sha256(rows_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        analyzer,
        "EXPECTED_SUMMARY_SHA256",
        sha256(summary_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match=match):
        analyzer.analyze(
            summary_path=summary_path,
            rows_path=rows_path,
            corpus_path=CORPUS_PATH,
        )


def test_conclusion_preserves_projection_and_claim_boundaries(tmp_path, monkeypatch):
    analyzer, summary_path, rows_path, _, _ = _fixture(tmp_path, monkeypatch)
    evidence = analyzer.analyze(
        summary_path=summary_path,
        rows_path=rows_path,
        corpus_path=CORPUS_PATH,
    )
    conclusion_path = tmp_path / "conclusion.md"

    analyzer.write_conclusion(evidence, conclusion_path)

    text = conclusion_path.read_text(encoding="utf-8")
    assert "restart：3/4" in text
    assert "31/32" in text
    assert "5816" in text and "5812" in text
    assert "safe L-BFGS：1/4" in text
    assert "29/32" in text
    assert "5370" in text and "5366" in text
    assert "FIRE：4/4" in text
    assert "32/32" in text
    assert "5590" in text and "5586" in text
    assert "490" in text and "9.6153846%" in text
    assert "cached terminal evaluation" in text
    assert "投影" in text and "不是当前实测" in text
    assert "不同 fallback endpoint energy" in text
    assert "不作速度公平比较" in text
    assert "不声称" in text and "默认" in text
    assert "ASE-LBFGS primary + certificate-triggered FIRE fallback" in text
    assert "固定预算 end-to-end" in text
    assert "不加入额外 fallback 层或参数" in text
