"""Fail-closed contracts for the proposal-fmax fixed-budget analyzer."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "20260728-proposal-fmax-end-to-end-gpu-ablation"
ANALYZER_PATH = RUN_ROOT / "analyze_ablation.py"
EXPECTED_COMMIT = "4ca77298d074429eb98a0238e6172d5e91ce9fb1"
FROZEN_HARNESS_PATH = str(
    REPO_ROOT
    / "runs"
    / "20260728-posterior-starter-policy-gpu-ablation"
    / "run_ablation.py"
)
FROZEN_HARNESS_SHA256 = "5168438843af122ee47c2fd9948cffcc6fca0ec15dbf2ad6ba1e9c8441665c41"
FROZEN_PAMSSW_BUNDLE_SHA256 = (
    "9ccab5f4ad38f368153448a13b277ed4786661fea9b5c15eb0091fff492d16d0"
)
FROZEN_INPUTS = {
    "c60": {
        "path": (
            "/mnt/d/download/trae-research-code/ssw/runs/"
            "20260428-c60-mace-production/prerelaxed_c60.xyz"
        ),
        "sha256": "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9",
    },
    "pdo": {
        "path": "/mnt/d/download/trae-research-code/ssw/PdO.xyz",
        "sha256": "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0",
    },
}
FROZEN_MODEL = {
    "path": "/root/.cache/mace/mace-omat-0-small.model",
    "sha256": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
}
SYSTEMS = ("c60", "pdo")
SEEDS = (42, 43, 44, 45)
ARMS = (
    ("proposal-fmax-0.05", 0.05),
    ("proposal-fmax-0.10", 0.10),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _config(system: str, proposal_fmax: float) -> dict[str, object]:
    return {
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": proposal_fmax,
        "proposal_pool_size": 1,
        "proposal_relax_steps": 80 if system == "c60" else 300,
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
        "quench_maxiter": 400,
        "rng_seed": 42,
    }


def _projection(system: str, arm_index: int, arm_id: str, proposal_fmax: float):
    preflight_case = (
        "/tmp/SSW-worktrees/safe-lbfgs-history-depth/"
        "runs/20260728-proposal-fmax-end-to-end-gpu-ablation/"
        f".preflight/{system}/{arm_id}"
    )
    return {
        "arm_index": arm_index,
        "arm_id": arm_id,
        "proposal_fmax": proposal_fmax,
        "source_config_type": "LSSSWConfig",
        "source_config": {
            "production": "frozen",
            "accepted_structures_dir": f"{preflight_case}/accepted_minima",
            "accepted_structures_log": (
                f"{preflight_case}/accepted_structures.jsonl"
            ),
            "direction_diagnostics_path": f"{preflight_case}/direction_trace.jsonl",
        },
        "removed_ls_fields": ["local_softening_strength"],
        "frozen_posterior_overrides": {"frozen": "posterior-harness"},
        "frozen_posterior_effective_ssw_config": _config(system, 0.05),
        "effective_ssw_config": _config(system, proposal_fmax),
        "arm_overrides": {"proposal_fmax": proposal_fmax},
        "proposal_protocol": {
            "optimizer": "safe-lbfgs-total",
            "history_limit": 10,
            "proposal_pool_size": 1,
            "softening_enabled": False,
        },
        "true_quench_protocol": {
            "optimizer": "ase-lbfgs",
            "fallback_optimizer": "ase-fire",
            "fmax_eV_per_A": 0.01,
            "maxiter": 400,
        },
        "nonoperative_config_outputs": {
            "accepted_structures_log": None,
            "accepted_structures_dir": None,
            "write_proposal_minima": False,
            "proposal_minima_dir": None,
            "write_relaxation_trajectories": False,
            "relaxation_trajectory_dir": None,
            "direction_diagnostics_enabled": False,
            "direction_diagnostics_path": None,
            "direction_archive_enabled": False,
            "direction_archive_path": None,
        },
    }


def _action_counts() -> dict[str, int]:
    return {
        "bootstrap_true_quench": 0,
        "starter_true_quench": 2,
        "direction_oracle": 10,
        "escape_true_pes_check": 3,
        "biased_proposal_relax": 50,
        "landing_true_quench": 33,
        "post_relax_validation": 2,
        "unattributed": 0,
    }


def _derive_action_seed(master_seed: int, batch_id: int, slot_id: int = 0) -> int:
    def pair(left: int, right: int) -> int:
        total = left + right
        return total * (total + 1) // 2 + right

    return pair(pair(master_seed, batch_id), slot_id)


def _fixture(tmp_path: Path) -> Path:
    output_root = tmp_path / "final-output"
    projections = {
        system: [
            _projection(system, arm_index, arm_id, proposal_fmax)
            for arm_index, (arm_id, proposal_fmax) in enumerate(ARMS)
        ]
        for system in SYSTEMS
    }
    frozen_preflight = {
        "schema_version": 1,
        "execution_commit": EXPECTED_COMMIT,
        "pamssw_bundle_sha256": FROZEN_PAMSSW_BUNDLE_SHA256,
        "systems": list(SYSTEMS),
        "master_seeds": list(SEEDS),
        "policies": ["uniform", "posterior_proportional", "minimal_ucb"],
        "batch_size": 1,
        "max_workers": 1,
        "action_force_budget": 1000,
        "total_force_budget": 6000,
        "inputs": deepcopy(FROZEN_INPUTS),
        "model": deepcopy(FROZEN_MODEL),
        "calculator": {
            "default_dtype": "float32",
            "device": "cuda",
            "enable_cueq": False,
            "inference_precision": "float32",
        },
        "runtime_versions": {
            "python": "3.12.0",
            "numpy": "2.1.0",
            "scipy": "1.17.0",
            "ase": "3.25.0",
            "torch": "2.8.0",
            "mace": "0.3.14",
        },
        "cuda": {
            "available": True,
            "requested_device": "cuda",
            "runtime_version": "12.8",
            "device_name": "test GPU",
        },
        "calculator_reuse": {
            "bootstrap": "one uncached caller-thread calculator",
            "actions": "one worker-thread-local calculator reused serially",
            "cross_thread_calculator_sharing": False,
            "accounting": "each bootstrap/action is wrapped by its own EvalCounter",
        },
        "projections": {
            system: {"effective_ssw_config": _config(system, 0.05)}
            for system in SYSTEMS
        },
    }
    matrix: list[dict[str, object]] = []
    for system in SYSTEMS:
        for seed in SEEDS:
            for arm_index, (arm_id, proposal_fmax) in enumerate(ARMS):
                matrix.append(
                    {
                        "matrix_index": len(matrix),
                        "system": system,
                        "master_seed": seed,
                        "policy_name": "uniform",
                        "batch_size": 1,
                        "max_workers": 1,
                        "arm_index": arm_index,
                        "arm_id": arm_id,
                        "proposal_fmax": proposal_fmax,
                        "relative_case_directory": (
                            f"{system}/seed-{seed:08d}/{arm_id}"
                        ),
                    }
                )
    manifest = {
        "schema_version": 1,
        "execution_commit": EXPECTED_COMMIT,
        "frozen_posterior_harness": {
            "path": FROZEN_HARNESS_PATH,
            "sha256": FROZEN_HARNESS_SHA256,
            "preflight": frozen_preflight,
        },
        "systems": list(SYSTEMS),
        "master_seeds": list(SEEDS),
        "policy_name": "uniform",
        "policy_support": "full-support random baseline",
        "batch_size": 1,
        "max_workers": 1,
        "action_force_budget": 1000,
        "total_force_budget": 6000,
        "arms": [
            {
                "arm_index": arm_index,
                "arm_id": arm_id,
                "proposal_fmax": proposal_fmax,
            }
            for arm_index, (arm_id, proposal_fmax) in enumerate(ARMS)
        ],
        "proposal_optimizer": "safe-lbfgs-total",
        "safe_lbfgs_history_limit": 10,
        "softening_enabled": False,
        "projections": projections,
        "matrix": matrix,
    }
    campaigns: list[dict[str, object]] = []
    for row in matrix:
        system = str(row["system"])
        seed = int(row["master_seed"])
        arm_index = int(row["arm_index"])
        arm_id = str(row["arm_id"])
        proposal_fmax = float(row["proposal_fmax"])
        campaign_root = output_root / str(row["relative_case_directory"])
        action_path = campaign_root / "action_metrics.jsonl"
        event_path = campaign_root / "events.jsonl"
        summary_path = campaign_root / "campaign_summary.json"
        bootstrap = 49 if system == "c60" else 84
        bootstrap_energy = -10.0 - seed * 0.01 - arm_index * 0.001
        best_energy = bootstrap_energy - 1.0 - arm_index * 0.1
        counts = _action_counts()
        force_evaluations = sum(counts.values())
        first = {
            "schema_version": 1,
            "ordinal": 0,
            "action_id": "batch-00000000-slot-0000",
            "batch_id": 0,
            "starter_id": 0,
            "selection_probability": 1.0,
            "policy_name": "uniform",
            "policy_support_complete": True,
            "eligible_starter_ids": [0],
            "probabilities": [1.0],
            "posterior_before": {"successes": 0, "failures": 0, "mean": 0.5},
            "status": "completed",
            "failure_reason": None,
            "posterior_observed": True,
            "discovered_against_snapshot": True,
            "inserted_into_archive": True,
            "landing_energy_eV": best_energy,
            "best_landing_energy_eV": best_energy,
            "unique_minima": 2,
            "evaluation_counts": deepcopy(counts),
            "evaluator_calls": force_evaluations,
            "force_evaluations": force_evaluations,
            "cumulative_action_force_evaluations": force_evaluations,
            "evaluator_wall_time_s": 1.0,
        }
        second = {
            **deepcopy(first),
            "ordinal": 1,
            "action_id": "batch-00000001-slot-0000",
            "batch_id": 1,
            "eligible_starter_ids": [0, 1],
            "probabilities": [0.5, 0.5],
            "selection_probability": 0.5,
            "posterior_before": {"successes": 1, "failures": 0, "mean": 2.0 / 3.0},
            "discovered_against_snapshot": False,
            "inserted_into_archive": False,
            "landing_energy_eV": best_energy,
            "best_landing_energy_eV": best_energy,
            "unique_minima": 2,
            "cumulative_action_force_evaluations": 2 * force_evaluations,
            "evaluator_wall_time_s": 1.1,
        }
        _write_jsonl(action_path, [first, second])
        events = [
            {
                "schema_version": 2,
                "record_type": "policy_snapshot",
                "archive_version": 0,
                "batch_id": 0,
                "eligible_starter_ids": [0],
                "policy_name": "uniform",
                "policy_version": 0,
                "probabilities": [1.0],
                "support_complete": True,
            },
            {
                "schema_version": 2,
                "record_type": "attempt",
                "action_id": first["action_id"],
                "archive_version": 0,
                "batch_id": 0,
                "cost_is_exact": True,
                "discovered_against_snapshot": True,
                "evaluation_counts": deepcopy(counts),
                "failure_reason": None,
                "force_budget": 1000,
                "force_evaluations": force_evaluations,
                "inserted_into_archive": True,
                "landing_energy": best_energy,
                "landing_entry_id": 1,
                "policy_name": "uniform",
                "policy_version": 0,
                "posterior_observed": True,
                "random_seed": _derive_action_seed(seed, 0),
                "selection_probability": 1.0,
                "slot_id": 0,
                "starter_id": 0,
                "status": "completed",
                "within_batch_collision": False,
            },
            {
                "schema_version": 2,
                "record_type": "batch_commit",
                "action_ids": [first["action_id"]],
                "batch_id": 0,
            },
            {
                "schema_version": 2,
                "record_type": "policy_snapshot",
                "archive_version": 1,
                "batch_id": 1,
                "eligible_starter_ids": [0, 1],
                "policy_name": "uniform",
                "policy_version": 1,
                "probabilities": [0.5, 0.5],
                "support_complete": True,
            },
            {
                "schema_version": 2,
                "record_type": "attempt",
                "action_id": second["action_id"],
                "archive_version": 1,
                "batch_id": 1,
                "cost_is_exact": True,
                "discovered_against_snapshot": False,
                "evaluation_counts": deepcopy(counts),
                "failure_reason": None,
                "force_budget": 1000,
                "force_evaluations": force_evaluations,
                "inserted_into_archive": False,
                "landing_energy": best_energy,
                "landing_entry_id": 1,
                "policy_name": "uniform",
                "policy_version": 1,
                "posterior_observed": True,
                "random_seed": _derive_action_seed(seed, 1),
                "selection_probability": 0.5,
                "slot_id": 0,
                "starter_id": 0,
                "status": "completed",
                "within_batch_collision": False,
            },
            {
                "schema_version": 2,
                "record_type": "batch_commit",
                "action_ids": [second["action_id"]],
                "batch_id": 1,
            },
        ]
        _write_jsonl(event_path, events)
        fallback_attempts = int(system == "pdo" and seed == 43 and arm_index == 1)
        _write_json(
            campaign_root / "optimizer_diagnostics.json",
            {
                "schema_version": 1,
                "attempts": [
                    {
                        "action_id": action["action_id"],
                        "stats": {
                            "budget_exhausted": 0,
                            "diagnostic_stage": "completed",
                            "force_evaluations": force_evaluations,
                            "proposal_optimizer": "safe-lbfgs-total",
                            "quench_optimizer": "ase-lbfgs",
                            "quench_fallback_optimizer": "ase-fire",
                            "quench_fallback_attempts": fallback_attempts,
                            "quench_fallback_converged": fallback_attempts,
                            "true_quench_unconverged": 0,
                        },
                    }
                    for action in (first, second)
                ],
            },
        )
        purpose_counts = {
            purpose: 2 * counts[purpose] for purpose in PURPOSES
        }
        purpose_counts["bootstrap_true_quench"] = bootstrap - 1
        purpose_counts["post_relax_validation"] += 1
        total = bootstrap + 2 * force_evaluations
        _write_json(
            summary_path,
            {
                "schema_version": 1,
                "policy_name": "uniform",
                "batch_size": 1,
                "max_workers": 1,
                "master_seed": seed,
                "action_force_budget": 1000,
                "total_force_budget": 6000,
                "bootstrap_evaluations": bootstrap,
                "action_evaluations": 2 * force_evaluations,
                "total_evaluations": total,
                "unused_force_budget": 6000 - total,
                "purpose_counts": purpose_counts,
                "completed_batches": 2,
                "completed_attempts": 2,
                "failed_attempts": 0,
                "posterior_observed_attempts": 2,
                "benchmark_eligible": True,
                "benchmark_ineligibility_reasons": [],
                "event_log_replayed": True,
                "archive_entries": 2,
                "best_archive_energy_eV": best_energy,
                "bootstrap_energy_eV": bootstrap_energy,
                "duplicate_rate": 1.0 / 3.0,
                "final_posterior": [
                    {
                        "starter_id": 0,
                        "energy_eV": bootstrap_energy,
                        "successes": 1,
                        "failures": 1,
                        "mean": 0.5,
                    },
                    {
                        "starter_id": 1,
                        "energy_eV": best_energy,
                        "successes": 0,
                        "failures": 0,
                        "mean": 0.5,
                    },
                ],
                "campaign_wall_time_s": 2.5 + arm_index,
                "bootstrap_evaluator_wall_time_s": 0.5,
                "action_metrics_path": str(action_path),
                "event_log_path": str(event_path),
                "calculator_factory": {
                    "bootstrap_instances": 1,
                    "action_instances": 1,
                    "action_thread_count": 1,
                },
            },
        )
        runtime_projection = deepcopy(projections[system][arm_index])
        for identity_field in ("arm_index", "arm_id", "proposal_fmax"):
            runtime_projection.pop(identity_field)
        runtime_case = (
            "runs/20260728-proposal-fmax-end-to-end-gpu-ablation/"
            f".output-seeds42-45-budget6000.partial/{row['relative_case_directory']}"
        )
        runtime_projection["source_config"].update(
            {
                "accepted_structures_dir": f"{runtime_case}/accepted_minima",
                "accepted_structures_log": (
                    f"{runtime_case}/accepted_structures.jsonl"
                ),
                "direction_diagnostics_path": (
                    f"{runtime_case}/direction_trace.jsonl"
                ),
            }
        )
        campaigns.append(
            {
                **row,
                "projection": runtime_projection,
                "campaign_summary_path": str(summary_path),
            }
        )
    _write_json(output_root / "manifest.json", manifest)
    _write_json(
        output_root / "index.json",
        {"schema_version": 1, "manifest": manifest, "campaigns": campaigns},
    )
    return output_root


def _mutate_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)


def _mutate_jsonl(path: Path, mutate) -> None:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    mutate(values)
    _write_jsonl(path, values)


def test_output_path_resolver_uses_repo_root_for_relative_records(
    tmp_path: Path, monkeypatch
):
    analyzer = _load_analyzer("proposal_fmax_analyzer_repo_relative_paths")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "runs" / "final-output"
    artifact = output_root / "c60" / "campaign_summary.json"
    outside = repo_root / "outside.json"
    _write_json(artifact, {"schema_version": 1})
    _write_json(outside, {"schema_version": 1})
    monkeypatch.setattr(analyzer, "REPO_ROOT", repo_root.resolve())

    relative_record = str(artifact.relative_to(repo_root))
    assert analyzer._inside_output(
        relative_record,
        output_root=output_root.resolve(),
        label="relative artifact",
    ) == artifact.resolve()
    assert analyzer._inside_output(
        str(artifact.resolve()),
        output_root=output_root.resolve(),
        label="absolute artifact",
    ) == artifact.resolve()
    with pytest.raises(ValueError, match="escapes final output"):
        analyzer._inside_output(
            str(outside.relative_to(repo_root)),
            output_root=output_root.resolve(),
            label="outside artifact",
        )


def test_analyzer_requires_the_complete_16_campaign_matrix_and_describes_pairs(
    tmp_path: Path,
):
    analyzer = _load_analyzer("proposal_fmax_analyzer_happy")
    evidence = analyzer.analyze(output_root=_fixture(tmp_path))

    assert evidence["validation"]["status"] == "passed"
    assert evidence["validation"]["campaign_count"] == 16
    assert evidence["validation"]["matrix"] == {
        "systems": list(SYSTEMS),
        "master_seeds": list(SEEDS),
        "arms": [arm_id for arm_id, _ in ARMS],
    }
    assert evidence["overall"]["failed_attempts"] == 0
    assert evidence["overall"]["quench_fallback_attempts"] == 2
    assert evidence["campaigns"]["c60"]["42"]["proposal-fmax-0.05"][
        "total_force_evaluations"
    ] == 249
    paired = evidence["aggregates"]["paired_loose_minus_strict"]["pdo"]
    assert paired["seeds"] == list(SEEDS)
    assert paired["descriptive"]["campaign_wall_time_s"]["sign_counts"] == {
        "negative": 0,
        "zero": 0,
        "positive": 4,
    }
    assert evidence["claim_boundary"]["default_decision"] == "not established"
    conclusion = analyzer.render_conclusion(evidence)
    assert "16/16" in conclusion
    assert "small-n" in conclusion
    assert "GPU nondeterminism" in conclusion
    assert "p-value" in conclusion
    assert "Purpose-ledger FE mean/median" in conclusion


def test_analyzer_rejects_unknown_runtime_projection_scientific_drift(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_runtime_projection_drift")
    output_root = _fixture(tmp_path)
    _mutate_json(
        output_root / "index.json",
        lambda value: value["campaigns"][0]["projection"]["source_config"].__setitem__(
            "production", "drifted"
        ),
    )

    with pytest.raises(ValueError, match="runtime projection scientific mismatch"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_runtime_projection_path_with_wrong_suffix(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_runtime_projection_path")
    output_root = _fixture(tmp_path)
    _mutate_json(
        output_root / "index.json",
        lambda value: value["campaigns"][0]["projection"]["source_config"].__setitem__(
            "accepted_structures_log", "runtime/case/wrong-name.jsonl"
        ),
    )

    with pytest.raises(ValueError, match="source path suffix mismatch"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_campaign_index_arm_identity_drift(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_index_arm_identity")
    output_root = _fixture(tmp_path)
    _mutate_json(
        output_root / "index.json",
        lambda value: value["campaigns"][0].__setitem__(
            "arm_id", "proposal-fmax-wrong"
        ),
    )

    with pytest.raises(ValueError, match="campaign index arm identity mismatch"):
        analyzer.analyze(output_root=output_root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda root: _mutate_json(
                root / "manifest.json",
                lambda value: value.__setitem__("execution_commit", "0" * 40),
            ),
            "execution commit",
        ),
        (
            lambda root: _mutate_json(
                root / "index.json",
                lambda value: value["manifest"]["frozen_posterior_harness"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            "index/manifest",
        ),
        (
            lambda root: _mutate_json(
                root / "manifest.json",
                lambda value: value["projections"]["c60"][0][
                    "effective_ssw_config"
                ].__setitem__("quench_fmax", 0.02),
            ),
            "effective optimizer config",
        ),
        (
            lambda root: _mutate_json(
                root / "index.json",
                lambda value: value["campaigns"][0].__setitem__(
                    "campaign_summary_path", str(root.parent / "outside.json")
                ),
            ),
            "escapes final output",
        ),
        (
            lambda root: _mutate_json(
                root
                / "c60"
                / "seed-00000042"
                / "proposal-fmax-0.05"
                / "campaign_summary.json",
                lambda value: value.__setitem__(
                    "total_evaluations", value["total_evaluations"] + 1
                ),
            ),
            "total FE ledger",
        ),
    ),
)
def test_analyzer_fails_closed_on_provenance_config_path_or_fe_corruption(
    tmp_path: Path, mutation, match: str
):
    analyzer = _load_analyzer(f"proposal_fmax_analyzer_corrupt_{match}")
    output_root = _fixture(tmp_path)
    mutation(output_root)

    with pytest.raises(ValueError, match=match):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_nonuniform_multistarter_snapshot(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_uniform")
    output_root = _fixture(tmp_path)
    campaign = output_root / "c60" / "seed-00000042" / "proposal-fmax-0.05"
    _mutate_jsonl(
        campaign / "action_metrics.jsonl",
        lambda values: values[1].update(
            {"probabilities": [0.6, 0.4], "selection_probability": 0.6}
        ),
    )
    _mutate_jsonl(
        campaign / "events.jsonl",
        lambda values: (
            values[3].__setitem__("probabilities", [0.6, 0.4]),
            values[4].__setitem__("selection_probability", 0.6),
        ),
    )

    with pytest.raises(ValueError, match="uniform probabilities"):
        analyzer.analyze(output_root=output_root)


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda manifest: manifest["frozen_posterior_harness"]["preflight"].__setitem__(
                "pamssw_bundle_sha256", "f" * 64
            ),
            "PAMSSW bundle SHA mismatch",
        ),
        (
            lambda manifest: manifest["frozen_posterior_harness"]["preflight"][
                "inputs"
            ]["c60"].__setitem__("sha256", "e" * 64),
            "c60 input SHA mismatch",
        ),
        (
            lambda manifest: manifest["frozen_posterior_harness"]["preflight"][
                "inputs"
            ]["c60"].__setitem__("path", "/alternative/c60.xyz"),
            "c60 input path mismatch",
        ),
        (
            lambda manifest: manifest["frozen_posterior_harness"]["preflight"][
                "model"
            ].__setitem__("sha256", "d" * 64),
            "model SHA mismatch",
        ),
        (
            lambda manifest: manifest["frozen_posterior_harness"]["preflight"][
                "model"
            ].__setitem__("path", "/alternative/model"),
            "model path mismatch",
        ),
        (
            lambda manifest: manifest["frozen_posterior_harness"].__setitem__(
                "sha256", "c" * 64
            ),
            "frozen harness SHA mismatch",
        ),
        (
            lambda manifest: manifest["frozen_posterior_harness"].__setitem__(
                "path", "/alternative/frozen_harness.py"
            ),
            "frozen harness path mismatch",
        ),
    ),
)
def test_analyzer_rejects_valid_but_unpinned_provenance_in_both_manifest_copies(
    tmp_path: Path, mutate, match: str
):
    analyzer = _load_analyzer(f"proposal_fmax_analyzer_pinned_{match}")
    output_root = _fixture(tmp_path)
    _mutate_json(output_root / "manifest.json", mutate)
    _mutate_json(output_root / "index.json", lambda value: mutate(value["manifest"]))

    with pytest.raises(ValueError, match=match):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_synchronized_common_drift_from_frozen_config(
    tmp_path: Path,
):
    analyzer = _load_analyzer("proposal_fmax_analyzer_common_config_drift")
    output_root = _fixture(tmp_path)

    def drift_manifest(manifest):
        for projection in manifest["projections"]["c60"]:
            projection["effective_ssw_config"]["proposal_relax_steps"] = 999

    _mutate_json(output_root / "manifest.json", drift_manifest)

    def drift_index(index):
        drift_manifest(index["manifest"])
        for campaign in index["campaigns"]:
            if campaign["system"] == "c60":
                campaign["projection"]["effective_ssw_config"][
                    "proposal_relax_steps"
                ] = 999

    _mutate_json(output_root / "index.json", drift_index)

    with pytest.raises(ValueError, match="frozen posterior config"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_binds_outer_frozen_config_to_nested_preflight_projection(
    tmp_path: Path,
):
    analyzer = _load_analyzer("proposal_fmax_analyzer_nested_frozen_binding")
    output_root = _fixture(tmp_path)

    def drift_outer_projections(manifest):
        for projection in manifest["projections"]["c60"]:
            projection["frozen_posterior_effective_ssw_config"][
                "proposal_relax_steps"
            ] = 999
            projection["effective_ssw_config"]["proposal_relax_steps"] = 999

    _mutate_json(output_root / "manifest.json", drift_outer_projections)

    def drift_index(index):
        drift_outer_projections(index["manifest"])
        for campaign in index["campaigns"]:
            if campaign["system"] == "c60":
                campaign["projection"]["frozen_posterior_effective_ssw_config"][
                    "proposal_relax_steps"
                ] = 999
                campaign["projection"]["effective_ssw_config"][
                    "proposal_relax_steps"
                ] = 999

    _mutate_json(output_root / "index.json", drift_index)

    with pytest.raises(ValueError, match="nested frozen preflight config"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_event_action_outcome_misalignment(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_outcome")
    output_root = _fixture(tmp_path)
    _mutate_jsonl(
        output_root
        / "c60"
        / "seed-00000042"
        / "proposal-fmax-0.05"
        / "events.jsonl",
        lambda values: values[1].__setitem__("landing_energy", -99.0),
    )

    with pytest.raises(ValueError, match="event/action outcome"):
        analyzer.analyze(output_root=output_root)


@pytest.mark.parametrize(
    "corruption",
    ("policy_version", "archive_version", "slot_id", "random_seed", "collision"),
)
def test_analyzer_rejects_event_action_identity_corruption(
    tmp_path: Path, corruption: str
):
    analyzer = _load_analyzer(f"proposal_fmax_analyzer_event_identity_{corruption}")
    output_root = _fixture(tmp_path)
    event_path = (
        output_root
        / "c60"
        / "seed-00000042"
        / "proposal-fmax-0.05"
        / "events.jsonl"
    )

    def mutate(values):
        if corruption in {"policy_version", "archive_version"}:
            values[3][corruption] = 99
            values[4][corruption] = 99
        elif corruption == "collision":
            values[4]["within_batch_collision"] = True
        else:
            values[4][corruption] += 1

    _mutate_jsonl(event_path, mutate)

    with pytest.raises(ValueError, match="event/action identity"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_noncanonical_duplicate_action_identity(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_duplicate_action")
    output_root = _fixture(tmp_path)
    campaign = output_root / "c60" / "seed-00000042" / "proposal-fmax-0.05"
    _mutate_jsonl(
        campaign / "action_metrics.jsonl",
        lambda values: values[1].__setitem__("action_id", values[0]["action_id"]),
    )
    _mutate_jsonl(
        campaign / "events.jsonl",
        lambda values: (
            values[4].__setitem__("action_id", values[1]["action_id"]),
            values[5].__setitem__("action_ids", [values[1]["action_id"]]),
        ),
    )
    _mutate_json(
        campaign / "optimizer_diagnostics.json",
        lambda value: value["attempts"][1].__setitem__(
            "action_id", value["attempts"][0]["action_id"]
        ),
    )

    with pytest.raises(ValueError, match="canonical action identity"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_inconsistent_running_best_landing_energy(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_best_landing")
    output_root = _fixture(tmp_path)
    _mutate_jsonl(
        output_root
        / "c60"
        / "seed-00000042"
        / "proposal-fmax-0.05"
        / "action_metrics.jsonl",
        lambda values: values[1].__setitem__(
            "best_landing_energy_eV", values[1]["best_landing_energy_eV"] + 0.5
        ),
    )

    with pytest.raises(ValueError, match="best landing energy"):
        analyzer.analyze(output_root=output_root)


@pytest.mark.parametrize("replacement", ("2.5", True))
def test_analyzer_rejects_non_numeric_finite_fields(
    tmp_path: Path, replacement: object
):
    analyzer = _load_analyzer(f"proposal_fmax_analyzer_strict_number_{replacement}")
    output_root = _fixture(tmp_path)
    _mutate_json(
        output_root
        / "c60"
        / "seed-00000042"
        / "proposal-fmax-0.05"
        / "campaign_summary.json",
        lambda summary: summary.__setitem__("campaign_wall_time_s", replacement),
    )

    with pytest.raises(ValueError, match="JSON number"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_rejects_missing_paired_arm_and_partial_root(tmp_path: Path):
    analyzer = _load_analyzer("proposal_fmax_analyzer_missing_pair")
    output_root = _fixture(tmp_path)
    partial_link = tmp_path / ".linked-output.partial"
    partial_link.symlink_to(output_root, target_is_directory=True)

    with pytest.raises(ValueError, match="refusing to analyze a partial"):
        analyzer.analyze(output_root=partial_link)

    partial_target = tmp_path / ".actual-output.partial"
    partial_target.mkdir()
    benign_link = tmp_path / "benign-output-link"
    benign_link.symlink_to(partial_target, target_is_directory=True)
    with pytest.raises(ValueError, match="refusing to analyze a partial"):
        analyzer.analyze(output_root=benign_link)

    _mutate_json(
        output_root / "index.json", lambda value: value["campaigns"].pop()
    )

    with pytest.raises(ValueError, match="campaign matrix"):
        analyzer.analyze(output_root=output_root)

    partial_root = tmp_path / ".output.partial"
    partial_root.mkdir()
    with pytest.raises(ValueError, match="refusing to analyze a partial"):
        analyzer.analyze(output_root=partial_root)
