"""Contracts for the posterior starter-policy GPU smoke analyzer."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "20260728-posterior-starter-policy-gpu-ablation"
ANALYZER_PATH = RUN_ROOT / "analyze_smoke.py"
EXPECTED_COMMIT = "83d34b568246a811a45a77e5c04e022fea3df598"
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
PAMSSW_BUNDLE_SHA256 = "9ccab5f4ad38f368153448a13b277ed4786661fea9b5c15eb0091fff492d16d0"
INPUT_SHA256 = {
    "c60": "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9",
    "pdo": "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0",
}
SYSTEMS = ("c60", "pdo")
POLICIES = ("uniform", "posterior_proportional", "minimal_ucb")
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
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _effective_config(system: str) -> dict[str, object]:
    return {
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
        "quench_maxiter": 400,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "proposal_relax_steps": 80 if system == "c60" else 300,
        "proposal_pool_size": 1,
        "rng_seed": 42,
    }


def _fixture(tmp_path: Path) -> Path:
    output_root = tmp_path / "output"
    projections = {
        system: {
            "effective_ssw_config": _effective_config(system),
            "overrides": {
                "quench_optimizer": "ase-lbfgs",
                "quench_fallback_optimizer": "ase-fire",
                "quench_fmax": 0.01,
                "proposal_pool_size": 1,
            },
            "removed_ls_fields": ["local_softening_mode"],
            "softening_enabled": False,
            "source_config": {},
            "source_config_type": "LSSSWConfig",
        }
        for system in SYSTEMS
    }
    manifest = {
        "schema_version": 1,
        "execution_commit": EXPECTED_COMMIT,
        "systems": list(SYSTEMS),
        "policies": list(POLICIES),
        "master_seeds": [42],
        "action_force_budget": 1000,
        "total_force_budget": 6000,
        "batch_size": 1,
        "max_workers": 1,
        "calculator": {
            "default_dtype": "float32",
            "device": "cuda",
            "enable_cueq": False,
            "inference_precision": "float32",
        },
        "calculator_reuse": {
            "accounting": "each bootstrap/action is wrapped by its own EvalCounter",
            "actions": "one worker-thread-local calculator reused serially",
            "bootstrap": "one uncached caller-thread calculator",
            "cross_thread_calculator_sharing": False,
        },
        "cuda": {
            "available": True,
            "device_name": "test GPU",
            "requested_device": "cuda",
            "runtime_version": "12.8",
        },
        "inputs": {
            "c60": {"path": "/inputs/c60.xyz", "sha256": INPUT_SHA256["c60"]},
            "pdo": {"path": "/inputs/pdo.xyz", "sha256": INPUT_SHA256["pdo"]},
        },
        "model": {"path": "/model", "sha256": MODEL_SHA256},
        "pamssw_bundle_sha256": PAMSSW_BUNDLE_SHA256,
        "runtime_versions": {
            "ase": "3.25.0",
            "mace": "0.3.14",
            "numpy": "2.1.3",
            "python": "3.12.12",
            "scipy": "1.17.1",
            "torch": "2.8.0",
        },
        "projections": projections,
    }
    campaigns: list[dict[str, object]] = []
    for system in SYSTEMS:
        bootstrap_evaluations = 49 if system == "c60" else 84
        for policy_index, policy in enumerate(POLICIES):
            campaign_root = output_root / system / "seed-00000042" / policy
            campaign_root.mkdir(parents=True)
            action_id = "batch-00000000-slot-0000"
            support_complete = policy != "minimal_ucb"
            action_counts = {purpose: 0 for purpose in PURPOSES}
            action_counts.update(
                {
                    "starter_true_quench": 2,
                    "direction_oracle": 10,
                    "escape_true_pes_check": 3,
                    "biased_proposal_relax": 50,
                    "landing_true_quench": 33,
                    "post_relax_validation": 2,
                }
            )
            action_evaluations = sum(action_counts.values())
            summary_counts = deepcopy(action_counts)
            summary_counts["bootstrap_true_quench"] = bootstrap_evaluations - 1
            summary_counts["post_relax_validation"] += 1
            total_evaluations = bootstrap_evaluations + action_evaluations
            bootstrap_energy = -10.0 - policy_index * 0.01
            best_energy = bootstrap_energy - 1.0 - policy_index
            action_metric = {
                "schema_version": 1,
                "action_id": action_id,
                "batch_id": 0,
                "ordinal": 0,
                "policy_name": policy,
                "policy_support_complete": support_complete,
                "eligible_starter_ids": [0],
                "probabilities": [1.0],
                "selection_probability": 1.0,
                "starter_id": 0,
                "status": "completed",
                "failure_reason": None,
                "posterior_before": {"failures": 0, "mean": 0.5, "successes": 0},
                "posterior_observed": True,
                "discovered_against_snapshot": True,
                "inserted_into_archive": True,
                "landing_energy_eV": best_energy,
                "best_landing_energy_eV": best_energy,
                "unique_minima": 2,
                "evaluation_counts": action_counts,
                "evaluator_calls": action_evaluations,
                "force_evaluations": action_evaluations,
                "cumulative_action_force_evaluations": action_evaluations,
                "evaluator_wall_time_s": 1.0,
            }
            _write_jsonl(campaign_root / "action_metrics.jsonl", [action_metric])
            events = [
                {
                    "schema_version": 2,
                    "record_type": "policy_snapshot",
                    "archive_version": 0,
                    "batch_id": 0,
                    "eligible_starter_ids": [0],
                    "policy_name": policy,
                    "policy_version": 0,
                    "probabilities": [1.0],
                    "support_complete": support_complete,
                },
                {
                    "schema_version": 2,
                    "record_type": "attempt",
                    "action_id": action_id,
                    "archive_version": 0,
                    "batch_id": 0,
                    "cost_is_exact": True,
                    "discovered_against_snapshot": True,
                    "evaluation_counts": action_counts,
                    "failure_reason": None,
                    "force_budget": 1000,
                    "force_evaluations": action_evaluations,
                    "inserted_into_archive": True,
                    "landing_energy": best_energy,
                    "landing_entry_id": 1,
                    "policy_name": policy,
                    "policy_version": 0,
                    "posterior_observed": True,
                    "random_seed": 100 + policy_index,
                    "selection_probability": 1.0,
                    "slot_id": 0,
                    "starter_id": 0,
                    "status": "completed",
                    "within_batch_collision": False,
                },
                {
                    "schema_version": 2,
                    "record_type": "batch_commit",
                    "action_ids": [action_id],
                    "batch_id": 0,
                },
            ]
            _write_jsonl(campaign_root / "events.jsonl", events)
            fallback_attempts = int(system == "pdo" and policy == "uniform")
            _write_json(
                campaign_root / "optimizer_diagnostics.json",
                {
                    "schema_version": 1,
                    "attempts": [
                        {
                            "action_id": action_id,
                            "stats": {
                                "budget_exhausted": 0,
                                "diagnostic_stage": "completed",
                                "force_evaluations": action_evaluations,
                                "proposal_optimizer": "safe-lbfgs-total",
                                "quench_optimizer": "ase-lbfgs",
                                "quench_fallback_optimizer": "ase-fire",
                                "quench_fallback_attempts": fallback_attempts,
                                "quench_fallback_converged": fallback_attempts,
                                "true_quench_unconverged": 0,
                            },
                        }
                    ],
                },
            )
            summary_path = campaign_root / "campaign_summary.json"
            summary = {
                "schema_version": 1,
                "action_evaluations": action_evaluations,
                "action_force_budget": 1000,
                "action_metrics_path": str(campaign_root / "action_metrics.jsonl"),
                "archive_entries": 2,
                "batch_size": 1,
                "benchmark_eligible": True,
                "benchmark_ineligibility_reasons": [],
                "best_archive_energy_eV": best_energy,
                "bootstrap_energy_eV": bootstrap_energy,
                "bootstrap_evaluations": bootstrap_evaluations,
                "bootstrap_evaluator_wall_time_s": 0.5,
                "calculator_factory": {
                    "action_instances": 1,
                    "action_thread_count": 1,
                    "bootstrap_instances": 1,
                },
                "campaign_wall_time_s": 1.5,
                "completed_attempts": 1,
                "completed_batches": 1,
                "duplicate_rate": 0.0,
                "event_log_path": str(campaign_root / "events.jsonl"),
                "event_log_replayed": True,
                "failed_attempts": 0,
                "final_posterior": [
                    {
                        "energy_eV": bootstrap_energy,
                        "failures": 0,
                        "mean": 2.0 / 3.0,
                        "starter_id": 0,
                        "successes": 1,
                    },
                    {
                        "energy_eV": best_energy,
                        "failures": 0,
                        "mean": 0.5,
                        "starter_id": 1,
                        "successes": 0,
                    },
                ],
                "master_seed": 42,
                "max_workers": 1,
                "policy_name": policy,
                "posterior_observed_attempts": 1,
                "purpose_counts": summary_counts,
                "total_evaluations": total_evaluations,
                "total_force_budget": 6000,
                "unused_force_budget": 6000 - total_evaluations,
            }
            _write_json(summary_path, summary)
            campaigns.append(
                {
                    "campaign_summary_path": str(summary_path),
                    "master_seed": 42,
                    "policy_name": policy,
                    "projection": deepcopy(projections[system]),
                    "system": system,
                }
            )
    _write_json(output_root / "manifest.json", manifest)
    _write_json(
        output_root / "index.json",
        {"schema_version": 1, "manifest": deepcopy(manifest), "campaigns": campaigns},
    )
    return output_root


def _add_second_multistarter_attempt(output_root: Path, policy: str) -> Path:
    campaign_root = output_root / "c60" / "seed-00000042" / policy
    metrics_path = campaign_root / "action_metrics.jsonl"
    events_path = campaign_root / "events.jsonl"
    diagnostics_path = campaign_root / "optimizer_diagnostics.json"
    summary_path = campaign_root / "campaign_summary.json"

    metrics = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    first = metrics[0]
    if policy == "uniform":
        probabilities = [0.5, 0.5]
        selected_starter = 0
    elif policy == "posterior_proportional":
        probabilities = [4.0 / 7.0, 3.0 / 7.0]
        selected_starter = 0
    else:
        probabilities = [0.0, 1.0]
        selected_starter = 1
    action_id = "batch-00000001-slot-0000"
    second = deepcopy(first)
    second.update(
        {
            "action_id": action_id,
            "batch_id": 1,
            "ordinal": 1,
            "eligible_starter_ids": [0, 1],
            "probabilities": probabilities,
            "selection_probability": probabilities[selected_starter],
            "starter_id": selected_starter,
            "posterior_before": (
                {"failures": 0, "mean": 2.0 / 3.0, "successes": 1}
                if selected_starter == 0
                else {"failures": 0, "mean": 0.5, "successes": 0}
            ),
            "discovered_against_snapshot": False,
            "inserted_into_archive": False,
            "cumulative_action_force_evaluations": 2 * first["force_evaluations"],
        }
    )
    _write_jsonl(metrics_path, [first, second])

    event_values = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    event_values.extend(
        [
            {
                "schema_version": 2,
                "record_type": "policy_snapshot",
                "archive_version": 1,
                "batch_id": 1,
                "eligible_starter_ids": [0, 1],
                "policy_name": policy,
                "policy_version": 1,
                "probabilities": probabilities,
                "support_complete": policy != "minimal_ucb",
            },
            {
                "schema_version": 2,
                "record_type": "attempt",
                "action_id": action_id,
                "archive_version": 1,
                "batch_id": 1,
                "cost_is_exact": True,
                "discovered_against_snapshot": False,
                "evaluation_counts": first["evaluation_counts"],
                "failure_reason": None,
                "force_budget": 1000,
                "force_evaluations": first["force_evaluations"],
                "inserted_into_archive": False,
                "landing_energy": first["landing_energy_eV"],
                "landing_entry_id": 1,
                "policy_name": policy,
                "policy_version": 1,
                "posterior_observed": True,
                "random_seed": 200,
                "selection_probability": probabilities[selected_starter],
                "slot_id": 0,
                "starter_id": selected_starter,
                "status": "completed",
                "within_batch_collision": False,
            },
            {
                "schema_version": 2,
                "record_type": "batch_commit",
                "action_ids": [action_id],
                "batch_id": 1,
            },
        ]
    )
    _write_jsonl(events_path, event_values)

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    second_diagnostic = deepcopy(diagnostics["attempts"][0])
    second_diagnostic["action_id"] = action_id
    diagnostics["attempts"].append(second_diagnostic)
    _write_json(diagnostics_path, diagnostics)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    action_cost = first["force_evaluations"]
    summary["action_evaluations"] += action_cost
    summary["total_evaluations"] += action_cost
    summary["unused_force_budget"] -= action_cost
    summary["completed_attempts"] = 2
    summary["completed_batches"] = 2
    summary["posterior_observed_attempts"] = 2
    summary["duplicate_rate"] = 1.0 / 3.0
    for purpose, count in first["evaluation_counts"].items():
        summary["purpose_counts"][purpose] += count
    if selected_starter == 0:
        summary["final_posterior"][0].update(
            {"failures": 1, "mean": 0.5, "successes": 1}
        )
    else:
        summary["final_posterior"][1].update(
            {"failures": 1, "mean": 1.0 / 3.0, "successes": 0}
        )
    _write_json(summary_path, summary)
    return campaign_root


def test_analyzer_validates_complete_matrix_ledgers_and_policy_support(tmp_path: Path):
    analyzer = _load_analyzer("posterior_smoke_analyzer_happy")
    output_root = _fixture(tmp_path)

    evidence = analyzer.analyze(output_root=output_root)

    assert evidence["validation"]["status"] == "passed"
    assert evidence["validation"]["campaign_count"] == 6
    assert evidence["validation"]["execution_commit"] == EXPECTED_COMMIT
    assert evidence["overall"]["completed_attempts"] == 6
    assert evidence["overall"]["failed_attempts"] == 0
    assert evidence["overall"]["quench_fallback_attempts"] == 1
    assert evidence["overall"]["quench_fallback_converged"] == 1
    assert evidence["campaigns"]["c60"]["uniform"]["bootstrap_force_evaluations"] == 49
    assert evidence["campaigns"]["pdo"]["uniform"]["bootstrap_force_evaluations"] == 84
    assert evidence["campaigns"]["c60"]["posterior_proportional"][
        "best_energy_drop_eV"
    ] == pytest.approx(2.0)
    assert evidence["policy_semantics"]["uniform"]["support_complete"] is True
    assert evidence["policy_semantics"]["posterior_proportional"][
        "strictly_positive_on_all_eligible"
    ] is True
    assert evidence["policy_semantics"]["minimal_ucb"]["support_complete"] is False
    assert evidence["policy_semantics"]["minimal_ucb"]["one_hot"] is True
    assert (
        evidence["diagnostic_scope"]["true_quench_cost"]
        == "purpose_ledger_not_final_only_optimizer_diagnostics"
    )
    assert evidence["validation"]["event_field_validation"] == "passed"
    assert evidence["validation"]["policy_probability_reconstruction"] == "passed"
    assert evidence["validation"]["posterior_credit_reconstruction"] == "passed"
    assert "event_replay" not in evidence["validation"]
    assert evidence["claim_boundary"]["policy_ranking"] == "not established"


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
                lambda value: value["campaigns"].pop(),
            ),
            "campaign matrix",
        ),
        (
            lambda root: _mutate_json(
                root / "manifest.json",
                lambda value: value["projections"]["c60"][
                    "effective_ssw_config"
                ].__setitem__("quench_optimizer", "scipy-lbfgsb"),
            ),
            "effective optimizer protocol",
        ),
        (
            lambda root: _mutate_json(
                root
                / "c60"
                / "seed-00000042"
                / "uniform"
                / "campaign_summary.json",
                lambda value: value.__setitem__(
                    "total_evaluations", value["total_evaluations"] + 1
                ),
            ),
            "total evaluation ledger",
        ),
        (
            lambda root: _mutate_jsonl(
                root
                / "c60"
                / "seed-00000042"
                / "posterior_proportional"
                / "events.jsonl",
                lambda values: values[0].update(
                    {
                        "eligible_starter_ids": [0, 1],
                        "probabilities": [1.0, 0.0],
                    }
                ),
            ),
            "eligible starters",
        ),
        (
            lambda root: _mutate_json(
                root
                / "pdo"
                / "seed-00000042"
                / "uniform"
                / "optimizer_diagnostics.json",
                lambda value: value["attempts"][0]["stats"].__setitem__(
                    "true_quench_unconverged", 1
                ),
            ),
            "true quench convergence",
        ),
    ),
)
def test_analyzer_fails_closed_on_semantic_or_accounting_corruption(
    tmp_path: Path,
    mutation,
    match: str,
):
    analyzer = _load_analyzer(f"posterior_smoke_analyzer_failure_{match}")
    output_root = _fixture(tmp_path)
    mutation(output_root)

    with pytest.raises(ValueError, match=match):
        analyzer.analyze(output_root=output_root)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("discovered_against_snapshot", False),
        ("inserted_into_archive", False),
        ("landing_energy", -10.5),
    ),
)
def test_analyzer_rejects_attempt_event_outcome_corruption(
    tmp_path: Path,
    field: str,
    replacement: object,
):
    analyzer = _load_analyzer(f"posterior_smoke_outcome_{field}")
    output_root = _fixture(tmp_path)
    _mutate_jsonl(
        output_root
        / "c60"
        / "seed-00000042"
        / "uniform"
        / "events.jsonl",
        lambda values: values[1].__setitem__(field, replacement),
    )

    with pytest.raises(ValueError, match="attempt event/action metric"):
        analyzer.analyze(output_root=output_root)


@pytest.mark.parametrize("corruption", ("counts", "energy"))
def test_analyzer_rejects_final_posterior_corruption(
    tmp_path: Path,
    corruption: str,
):
    analyzer = _load_analyzer(f"posterior_smoke_posterior_{corruption}")
    output_root = _fixture(tmp_path)

    def mutate(summary):
        if corruption == "counts":
            summary["final_posterior"][0].update(
                {"successes": 0, "failures": 1, "mean": 1.0 / 3.0}
            )
        else:
            summary["final_posterior"][1]["energy_eV"] += 0.25

    _mutate_json(
        output_root
        / "c60"
        / "seed-00000042"
        / "uniform"
        / "campaign_summary.json",
        mutate,
    )

    with pytest.raises(ValueError, match="posterior reconstruction"):
        analyzer.analyze(output_root=output_root)


def test_analyzer_requires_index_manifest_projection_exactly_match_manifest(
    tmp_path: Path,
):
    analyzer = _load_analyzer("posterior_smoke_index_projection")
    output_root = _fixture(tmp_path)
    _mutate_json(
        output_root / "index.json",
        lambda index: index["manifest"]["projections"]["c60"][
            "effective_ssw_config"
        ].__setitem__("proposal_relax_steps", 999),
    )

    with pytest.raises(ValueError, match="index/manifest provenance"):
        analyzer.analyze(output_root=output_root)


@pytest.mark.parametrize("policy", POLICIES)
def test_analyzer_validates_true_multistarter_policy_snapshots(
    tmp_path: Path,
    policy: str,
):
    analyzer = _load_analyzer(f"posterior_smoke_multistarter_{policy}")
    output_root = _fixture(tmp_path)
    _add_second_multistarter_attempt(output_root, policy)

    evidence = analyzer.analyze(output_root=output_root)

    assert evidence["campaigns"]["c60"][policy]["completed_attempts"] == 2


@pytest.mark.parametrize(
    ("policy", "wrong_probabilities"),
    (
        ("uniform", [0.6, 0.4]),
        ("posterior_proportional", [0.5, 0.5]),
    ),
)
def test_analyzer_rejects_wrong_positive_multistarter_policy_weights(
    tmp_path: Path,
    policy: str,
    wrong_probabilities: list[float],
):
    analyzer = _load_analyzer(f"posterior_smoke_wrong_weights_{policy}")
    output_root = _fixture(tmp_path)
    campaign_root = _add_second_multistarter_attempt(output_root, policy)
    _mutate_jsonl(
        campaign_root / "action_metrics.jsonl",
        lambda values: values[1].update(
            {
                "probabilities": wrong_probabilities,
                "selection_probability": wrong_probabilities[0],
            }
        ),
    )
    _mutate_jsonl(
        campaign_root / "events.jsonl",
        lambda values: (
            values[3].__setitem__("probabilities", wrong_probabilities),
            values[4].__setitem__("selection_probability", wrong_probabilities[0]),
        ),
    )

    with pytest.raises(ValueError, match="policy probabilities"):
        analyzer.analyze(output_root=output_root)


def _mutate_json(path: Path, mutation) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(path, value)


def _mutate_jsonl(path: Path, mutation) -> None:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    mutation(values)
    _write_jsonl(path, values)


def test_conclusion_states_single_seed_and_bootstrap_diagnostic_boundaries(
    tmp_path: Path,
):
    analyzer = _load_analyzer("posterior_smoke_analyzer_conclusion")
    evidence = analyzer.analyze(output_root=_fixture(tmp_path))

    conclusion = analyzer.render_conclusion(evidence)

    assert "6/6" in conclusion
    assert "单个 seed" in conclusion
    assert "bootstrap fallback" in conclusion
    assert "frequency-unbiased" in conclusion
    assert "TS" in conclusion
