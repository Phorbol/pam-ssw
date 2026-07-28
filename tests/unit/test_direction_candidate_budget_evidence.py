from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ANALYZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-direction-candidate-budget-gpu-ablation"
    / "analyze_evidence.py"
)


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("direction_budget_evidence", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(module, *, system: str, seed: int, arm: str, drop: float) -> dict:
    output = f"output-{arm}-{system}-seed{seed}"
    oracle_candidates = 12 if system == "c60" else 8
    candidate_max = oracle_candidates if arm == "hardcap" else oracle_candidates + 1
    candidate_sum = 20
    direction_choices = 4
    commit = module.ARM_COMMITS[arm]
    return {
        "arm": arm,
        "system": system,
        "seed": seed,
        "total_force_budget": 6000,
        "force_evaluations": 6000,
        "initial_energy_eV": -1.0,
        "best_energy_eV": -1.0 - drop,
        "energy_drop_eV": drop,
        "base_preflight": {
            "execution_commit": commit,
            "model_sha256": module.MODEL_SHA256,
            "input_sha256": module.INPUT_SHA256[system],
            "runtime_versions": {"python": "test"},
            "cuda": {"available": True, "device_name": "test-gpu"},
            "calculator": {"device": "cuda", "default_dtype": "float32"},
        },
        "target": {
            "execution_commit": commit,
            "frozen_runner_sha256": "frozen-runner-sha",
        },
        "native_generator_contract": {
            "candidate_budget": 2,
            "expectation": (
                "nonfirst_native_candidates_exceed_budget"
                if arm == "precap"
                else "native_candidates_do_not_exceed_budget"
            ),
            "nonfirst_candidate_count": 4 if arm == "precap" else 2,
        },
        "purpose_counts": {
            "direction_oracle": 2 * candidate_sum,
            "biased_proposal_relax": 5960,
            "unattributed": 0,
        },
        "direction_oracle_audit": {
            "candidate_count_sum": candidate_sum,
            "candidate_count_max": candidate_max,
            "expected_force_evaluations": 2 * candidate_sum,
            "recorded_force_evaluations": 2 * candidate_sum,
        },
        "effective_config": {
            "oracle_candidates": oracle_candidates,
            "accepted_structures_dir": f"{output}/accepted_minima",
            "accepted_structures_log": f"{output}/accepted_structures.jsonl",
            "direction_diagnostics_path": f"{output}/direction_trace.jsonl",
            "frozen_parameter": 7,
        },
        "source_config": {
            "oracle_candidates": oracle_candidates,
            "accepted_structures_dir": f"{output}/accepted_minima",
            "accepted_structures_log": f"{output}/accepted_structures.jsonl",
            "direction_diagnostics_path": f"{output}/direction_trace.jsonl",
            "frozen_parameter": 7,
        },
        "stats": {
            "budget_exhausted": 1,
            "n_minima": 4,
            "n_trials": 5,
            "duplicate_rate": 0.25,
            "direction_choices": direction_choices,
            "direction_candidate_evaluations": candidate_sum,
            "direction_mean_candidate_pool_size": candidate_sum / direction_choices,
        },
        "record_counts": {
            "direction_records": direction_choices,
            "n_trials": 5,
            "unlogged_direction_choices": 0,
            "unlogged_walk_trials": 0,
            "walk_records": 5,
        },
        "timing": {"total_wall_time_s": 10.0},
    }


def _write_six_summaries(module, root: Path) -> None:
    for system, seed in module.PAIRS:
        for arm in ("precap", "hardcap"):
            drop = 2.0 if arm == "hardcap" else 1.0
            output = root / f"output-{arm}-{system}-seed{seed}"
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps(
                    _summary(
                        module,
                        system=system,
                        seed=seed,
                        arm=arm,
                        drop=drop,
                    )
                ),
                encoding="utf-8",
            )


def test_build_evidence_computes_paired_delta_and_rejects_unattributed(tmp_path):
    analyzer = _load_analyzer()
    _write_six_summaries(analyzer, tmp_path)

    evidence = analyzer.build_evidence(tmp_path)

    assert len(evidence["runs"]) == 6
    assert evidence["pairs"][0]["delta_hardcap_minus_precap"]["energy_drop_eV"] == 1.0
    assert evidence["scope"]["excluded_glob"] == "output-*.partial-failed"
    assert len(evidence["provenance"]["analyzer_sha256"]) == 64
    assert all(len(run["summary_sha256"]) == 64 for run in evidence["runs"])
    conclusion = analyzer.render_conclusion(evidence)
    assert "precap overflow" in conclusion
    assert "运行环境身份一致" in conclusion

    bad_summary = (
        tmp_path / "output-hardcap-c60-seed42" / "summary.json"
    )
    payload = json.loads(bad_summary.read_text(encoding="utf-8"))
    payload["purpose_counts"]["unattributed"] = 1
    bad_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(analyzer.EvidenceError, match="unattributed"):
        analyzer.build_evidence(tmp_path)


@pytest.mark.parametrize(
    ("run_name", "path", "value", "message"),
    [
        (
            "output-hardcap-c60-seed42",
            ("force_evaluations",),
            6000.0,
            "exact nonnegative integer",
        ),
        (
            "output-hardcap-c60-seed42",
            ("purpose_counts", "unattributed"),
            False,
            "exact nonnegative integer",
        ),
        (
            "output-hardcap-c60-seed42",
            ("stats", "n_minima"),
            "4",
            "exact nonnegative integer",
        ),
        (
            "output-hardcap-c60-seed42",
            ("direction_oracle_audit", "candidate_count_sum"),
            -1,
            "exact nonnegative integer",
        ),
        (
            "output-hardcap-c60-seed42",
            ("initial_energy_eV",),
            float("nan"),
            "finite",
        ),
        (
            "output-hardcap-c60-seed42",
            ("stats", "duplicate_rate"),
            1.1,
            "duplicate_rate",
        ),
        (
            "output-precap-c60-seed42",
            ("direction_oracle_audit", "candidate_count_max"),
            12,
            "precap overflow",
        ),
        (
            "output-hardcap-c60-seed42",
            ("native_generator_contract", "expectation"),
            "wrong",
            "native generator contract",
        ),
        (
            "output-hardcap-c60-seed42",
            ("record_counts", "direction_records"),
            5,
            "record counts",
        ),
        (
            "output-hardcap-c60-seed42",
            ("stats", "direction_candidate_evaluations"),
            21,
            "candidate evaluations",
        ),
        (
            "output-hardcap-c60-seed42",
            ("stats", "direction_mean_candidate_pool_size"),
            6.0,
            "candidate mean",
        ),
        (
            "output-hardcap-c60-seed42",
            ("base_preflight", "runtime_versions", "python"),
            "other",
            "paired runtime",
        ),
        (
            "output-hardcap-c60-seed42",
            ("base_preflight", "cuda", "available"),
            False,
            "GPU runtime",
        ),
        (
            "output-hardcap-c60-seed42",
            ("target", "frozen_runner_sha256"),
            "other",
            "paired runtime",
        ),
    ],
)
def test_build_evidence_rejects_contract_mutations(
    tmp_path, run_name, path, value, message
):
    analyzer = _load_analyzer()
    _write_six_summaries(analyzer, tmp_path)
    summary_path = tmp_path / run_name / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(analyzer.EvidenceError, match=message):
        analyzer.build_evidence(tmp_path)
