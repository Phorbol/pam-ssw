"""Contract tests for the fixed raw-landing strict-quench analyzer."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "20260728-true-quench-raw-strict-replay"
ANALYZER_PATH = RUN_ROOT / "analyze.py"
CORPUS_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-true-quench-tiered-ablation"
    / "output"
    / "corpus.json"
)
EXPECTED_COMMIT = "e3141c46b53e0a646fbe3da95678f04e389b4613"
CORPUS_SHA256 = "100759e1871cdefb54972f91751c452763f74d0e34ee576f268a5808219a552b"
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
ARMS = (
    ("scipy-lbfgsb", "scipy-lbfgsb", None),
    ("safe-lbfgs-total", "safe-lbfgs-total", 10),
    ("ase-fire", "ase-fire", None),
    ("ase-fire2", "ase-fire2", None),
    ("ase-lbfgs", "ase-lbfgs", None),
)
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


def _load_analyzer(name: str):
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    total_force_evaluations = 0
    for system in ("c60", "pdo"):
        entries = corpus["systems"][system]["entries"]
        for arm_index, (arm_id, optimizer, history_limit) in enumerate(ARMS):
            for task_index, entry in enumerate(entries):
                force_evaluations = 10 + arm_index + task_index
                total_force_evaluations += force_evaluations
                # Every arm has one certificate failure.  The optimizer flag is
                # deliberately true even on those rows.
                final_force = 0.011 if task_index == arm_index else 0.009
                purpose_counts = {purpose: 0 for purpose in PURPOSES}
                purpose_counts["landing_true_quench"] = force_evaluations
                rows.append(
                    {
                        "system": system,
                        "task_index": task_index,
                        "trial_index": task_index + 1,
                        "proposal_index": 1,
                        "source_trajectory_path": entry["source_trajectory_path"],
                        "source_trajectory_sha256": entry[
                            "source_trajectory_sha256"
                        ],
                        "source_frame_index": 0,
                        "arm_id": arm_id,
                        "optimizer": optimizer,
                        "safe_history_limit": history_limit,
                        "fmax_eV_per_A": 0.01,
                        "maxiter": 400,
                        "coordinate_trust_radius_A": None,
                        "objective": "true_mace_pes_no_bias_no_softening",
                        "initial": {
                            "energy_eV": -10.0,
                            "max_active_force_eV_per_A": 1.0,
                            "positions_sha256": entry["initial_positions_sha256"],
                            "state": deepcopy(entry["state"]),
                        },
                        "final": {
                            "energy_eV": -11.0 - 0.1 * task_index,
                            "max_active_force_eV_per_A": final_force,
                            "positions_sha256": entry["initial_positions_sha256"],
                            "state": deepcopy(entry["state"]),
                        },
                        "n_iter": force_evaluations - 1,
                        "termination_reason": (
                            "unconverged" if final_force > 0.01 else "converged"
                        ),
                        "outcome_class": (
                            "useful_progress"
                            if final_force > 0.01
                            else "converged_productive"
                        ),
                        "telemetry": {
                            "backend": optimizer,
                            "converged": final_force <= 0.01,
                            "optimizer_success": True,
                            "evaluator_calls": force_evaluations,
                            "gradient_measure": "raw_active_max_force",
                            "termination_reason": (
                                "unconverged"
                                if final_force > 0.01
                                else "converged"
                            ),
                        },
                        "evaluator_calls": force_evaluations,
                        "force_evaluations": force_evaluations,
                        "purpose_count_delta": purpose_counts,
                        "wall_time_s": 0.25 * force_evaluations,
                    }
                )
    summary_counts = {purpose: 0 for purpose in PURPOSES}
    summary_counts["landing_true_quench"] = total_force_evaluations
    summary = {
        "schema_version": 1,
        "execution_commit": EXPECTED_COMMIT,
        "row_count": 160,
        "calculator": {
            "default_dtype": "float32",
            "device": "cuda",
            "enable_cueq": False,
            "inference_precision": "float32",
        },
        "cuda": {
            "available": True,
            "device_name": "test GPU",
            "runtime_version": "12.8",
        },
        "runtime_versions": {
            "ase": "3.25.0",
            "python": "3.12.12",
        },
        "inputs": {
            "c60": {"path": "/inputs/c60.xyz", "sha256": "1" * 64},
            "pdo": {"path": "/inputs/pdo.xyz", "sha256": "2" * 64},
        },
        "corpus": {
            "path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
            "sha256": CORPUS_SHA256,
            "task_count_per_system": 16,
            "total_task_count": 32,
        },
        "model": {
            "path": "/model",
            "sha256": MODEL_SHA256,
        },
        "strict_protocol": {
            "fmax_eV_per_A": 0.01,
            "maxiter": 400,
            "objective": "true_mace_pes_no_bias_no_softening",
        },
        "arms": [
            {
                "arm_id": arm_id,
                "optimizer": optimizer,
                "safe_history_limit": history_limit,
            }
            for arm_id, optimizer, history_limit in ARMS
        ],
        "evaluation_counts": summary_counts,
        "rows_file": "rows.json",
    }
    summary_path = tmp_path / "summary.json"
    rows_path = tmp_path / "rows.json"
    _write_json(summary_path, summary)
    _write_json(rows_path, rows)
    return summary_path, rows_path, rows


def test_analyzer_uses_force_certificate_and_reports_complete_taskwise_coverage(
    tmp_path: Path,
):
    analyzer = _load_analyzer("raw_strict_analyzer_summary")
    summary_path, rows_path, _ = _fixture(tmp_path)

    evidence = analyzer.analyze(
        summary_path=summary_path,
        rows_path=rows_path,
        corpus_path=CORPUS_PATH,
    )

    assert evidence["validation"]["status"] == "passed"
    assert evidence["validation"]["row_count"] == 160
    assert evidence["validation"]["evaluation_counts"]["landing_true_quench"] > 0
    assert evidence["validation"]["evaluation_counts"]["unattributed"] == 0
    assert evidence["validation"]["provenance"] == {
        "calculator": {
            "default_dtype": "float32",
            "device": "cuda",
            "enable_cueq": False,
            "inference_precision": "float32",
        },
        "cuda": {
            "available": True,
            "device_name": "test GPU",
            "runtime_version": "12.8",
        },
        "runtime_versions": {
            "ase": "3.25.0",
            "python": "3.12.12",
        },
        "inputs": {
            "c60": {"path": "/inputs/c60.xyz", "sha256": "1" * 64},
            "pdo": {"path": "/inputs/pdo.xyz", "sha256": "2" * 64},
        },
        "model": {"path": "/model", "sha256": MODEL_SHA256},
    }
    assert (
        evidence["validation"]["cost_scope"]
        == "unconditional_all_attempts_including_successes_and_failures"
    )
    for system in ("c60", "pdo"):
        assert evidence["systems"][system]["taskwise_coverage"]["any_arm"] == 16
        assert evidence["systems"][system]["taskwise_coverage"]["all_arms"] == 11
        for arm_id, _, _ in ARMS:
            arm = evidence["systems"][system]["arms"][arm_id]
            assert arm["success_count"] == 15
            assert arm["success_rate"] == pytest.approx(15 / 16)
            assert arm["failure_count"] == 1
            assert arm["total_force_evaluations"] > 0
            assert arm["median_force_evaluations"] > 0
            assert arm["p90_force_evaluations"] >= arm["median_force_evaluations"]
            assert arm["max_force_evaluations"] >= arm["p90_force_evaluations"]
            assert arm["total_energy_drop_eV"] > 0
            assert arm["median_energy_drop_eV"] > 0
            assert arm["final_max_force_eV_per_A"]["max"] == pytest.approx(
                0.011
            )
    assert evidence["overall"]["single_arm_full_coverage"] == []
    assert evidence["claim_boundary"]["global_search_performance"] == "not measured"


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
            lambda summary, rows: summary["corpus"].__setitem__(
                "sha256", "0" * 64
            ),
            "corpus SHA",
        ),
        (
            lambda summary, rows: summary["corpus"].__setitem__(
                "path", "other-corpus.json"
            ),
            "corpus path",
        ),
        (
            lambda summary, rows: summary["model"].__setitem__(
                "sha256", "0" * 64
            ),
            "model SHA",
        ),
        (
            lambda summary, rows: rows.pop(),
            "row count",
        ),
        (
            lambda summary, rows: rows[0]["initial"].__setitem__(
                "positions_sha256", "0" * 64
            ),
            "raw identity",
        ),
        (
            lambda summary, rows: rows[0].__setitem__("fmax_eV_per_A", 0.02),
            "strict protocol",
        ),
        (
            lambda summary, rows: rows[0]["purpose_count_delta"].__setitem__(
                "unattributed", 1
            ),
            "evaluation ledger",
        ),
        (
            lambda summary, rows: rows[0].__setitem__(
                "task_index", rows[1]["task_index"]
            ),
            "duplicate",
        ),
        (
            lambda summary, rows: rows[0]["telemetry"].__setitem__(
                "converged", True
            ),
            "telemetry convergence",
        ),
        (
            lambda summary, rows: (
                rows[0].__setitem__("termination_reason", "converged"),
                rows[0]["telemetry"].__setitem__("termination_reason", "converged"),
                rows[0]["telemetry"].__setitem__("converged", True),
            ),
            "certificate convergence",
        ),
        (
            lambda summary, rows: summary["cuda"].__setitem__(
                "available", "yes"
            ),
            "provenance",
        ),
    ),
)
def test_analyzer_fails_closed_for_identity_protocol_matrix_and_ledger_errors(
    tmp_path: Path, mutation, match: str
):
    analyzer = _load_analyzer(f"raw_strict_analyzer_fail_{match.replace(' ', '_')}")
    summary_path, rows_path, rows = _fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mutation(summary, rows)
    _write_json(summary_path, summary)
    _write_json(rows_path, rows)

    with pytest.raises(ValueError, match=match):
        analyzer.analyze(
            summary_path=summary_path,
            rows_path=rows_path,
            corpus_path=CORPUS_PATH,
        )


def test_conclusion_states_evidence_and_claim_boundaries(tmp_path: Path):
    analyzer = _load_analyzer("raw_strict_analyzer_conclusion")
    summary_path, rows_path, _ = _fixture(tmp_path)
    evidence = analyzer.analyze(
        summary_path=summary_path,
        rows_path=rows_path,
        corpus_path=CORPUS_PATH,
    )
    conclusion_path = tmp_path / "conclusion.md"

    analyzer.write_conclusion(evidence, conclusion_path)

    text = conclusion_path.read_text(encoding="utf-8")
    assert "严格力证书" in text
    assert "不采信 optimizer_success" in text
    assert "成功与失败的全部 attempts" in text
    assert "unconverged:1" in text
    assert "没有单一优化器 arm 实现全覆盖" in text
    assert "不能据此声称全局搜索性能" in text
    assert "primary + fallback" in text
    assert "不直接修改默认值" in text
