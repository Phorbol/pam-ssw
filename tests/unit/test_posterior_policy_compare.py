import json
from pathlib import Path

import pytest

from benchmarks.posterior_policy_compare import main, run_comparison
from pamssw.accounting import EvaluationPurpose


EXPECTED_RECORD_FIELDS = {
    "schema_version",
    "calculator_label",
    "potential_parameters",
    "seed",
    "policy",
    "batch_size",
    "max_workers",
    "action_force_budget",
    "total_force_budget",
    "best_energy",
    "unique_minima",
    "completed_batches",
    "completed_attempts",
    "failed_attempts",
    "posterior_observed_attempts",
    "bootstrap_evaluations",
    "action_evaluations",
    "total_evaluations",
    "purpose_counts",
    "unused_force_budget",
    "stop_reason",
    "benchmark_eligible",
    "benchmark_ineligibility_reasons",
}

POLICIES = ("uniform", "posterior_proportional", "minimal_ucb")


def test_run_comparison_is_importable():
    assert callable(run_comparison)


def test_run_comparison_emits_paired_raw_facts_and_event_logs(tmp_path: Path):
    output_root = tmp_path / "paired-runs"

    records = run_comparison(
        output_root=output_root,
        seeds=(3, 7),
        policies=POLICIES,
        total_force_budget=120,
        action_force_budget=30,
        batch_size=2,
        max_workers=2,
    )

    assert [(record["seed"], record["policy"]) for record in records] == [
        (seed, policy) for seed in (3, 7) for policy in POLICIES
    ]
    assert len(records) == 6
    assert output_root.is_dir()

    for record in records:
        assert set(record) == EXPECTED_RECORD_FIELDS
        assert record["schema_version"] == 1
        assert record["calculator_label"] == "analytic-double-well-2d-v1"
        assert record["potential_parameters"] == {
            "energy_expression": "(x^2 - 1)^2 + 0.5*y^2 + 0.25*z^2",
            "x_well_positions": [-1.0, 1.0],
            "y_quadratic_coefficient": 0.5,
            "z_quadratic_coefficient": 0.25,
        }
        assert isinstance(record["best_energy"], float)
        assert isinstance(record["unique_minima"], int)
        assert record["unique_minima"] >= 1
        assert record["total_evaluations"] == (
            record["bootstrap_evaluations"] + record["action_evaluations"]
        )
        assert record["total_evaluations"] == sum(record["purpose_counts"].values())
        assert record["total_evaluations"] + record["unused_force_budget"] == 120
        assert record["total_evaluations"] <= 120
        assert set(record["purpose_counts"]) == {purpose.value for purpose in EvaluationPurpose}
        assert isinstance(record["stop_reason"], str)
        assert isinstance(record["benchmark_ineligibility_reasons"], list)
        assert {"winner", "ranking", "p_value", "score"}.isdisjoint(record)
        json.dumps(record, allow_nan=False)

        assert record["completed_batches"] >= 1
        event_path = output_root / f"seed-{record['seed']:08d}-{record['policy']}" / "events.jsonl"
        assert event_path.is_file()


def test_run_comparison_preserves_supplied_policy_order(tmp_path: Path):
    policies = ("minimal_ucb", "uniform")

    records = run_comparison(
        output_root=tmp_path / "ordered-runs",
        seeds=(5,),
        policies=policies,
        total_force_budget=120,
        action_force_budget=30,
        batch_size=2,
        max_workers=2,
    )

    assert [record["policy"] for record in records] == list(policies)


def test_run_comparison_rejects_existing_output_root_before_any_run(tmp_path: Path):
    output_root = tmp_path / "occupied-runs"
    output_root.mkdir()
    sentinel = output_root / "keep.txt"
    sentinel.write_text("do not touch", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output_root"):
        run_comparison(
            output_root=output_root,
            seeds=(3,),
            total_force_budget=120,
            action_force_budget=30,
            batch_size=2,
            max_workers=2,
        )

    assert sentinel.read_text(encoding="utf-8") == "do not touch"
    assert list(output_root.iterdir()) == [sentinel]


def test_cli_rejects_existing_output_or_derived_run_root(tmp_path: Path):
    output = tmp_path / "facts.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output"):
        main(
            [
                "--output",
                str(output),
                "--seeds",
                "3",
                "--total-force-budget",
                "120",
                "--action-force-budget",
                "30",
                "--batch-size",
                "2",
                "--max-workers",
                "2",
            ]
        )

    root_output = tmp_path / "root-facts.json"
    run_root = tmp_path / "root-facts-runs"
    run_root.mkdir()

    with pytest.raises(FileExistsError, match="output_root"):
        main(
            [
                "--output",
                str(root_output),
                "--seeds",
                "3",
                "--total-force-budget",
                "120",
                "--action-force-budget",
                "30",
                "--batch-size",
                "2",
                "--max-workers",
                "2",
            ]
        )


def test_cli_writes_parseable_raw_record_document(tmp_path: Path):
    output = tmp_path / "facts.json"

    assert main(
        [
            "--output",
            str(output),
            "--seeds",
            "3",
            "--total-force-budget",
            "120",
            "--action-force-budget",
            "30",
            "--batch-size",
            "2",
            "--max-workers",
            "2",
        ]
    ) == 0

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert isinstance(document["records"], list)
    assert len(document["records"]) == 3
    assert (tmp_path / "facts-runs").is_dir()
    assert output.read_text(encoding="utf-8").endswith("\n")
