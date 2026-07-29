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


def _attempt_metadata_by_slot(event_path: Path) -> dict[tuple[int, int], tuple[int, int]]:
    rows = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["record_type"] == "attempt"
    ]
    metadata = {
        (row["batch_id"], row["slot_id"]): (row["random_seed"], row["force_budget"])
        for row in rows
    }
    assert len(metadata) == len(rows)
    return metadata


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

    for seed in (3, 7):
        metadata_by_policy = [
            _attempt_metadata_by_slot(output_root / f"seed-{seed:08d}-{policy}" / "events.jsonl")
            for policy in POLICIES
        ]
        common_slots = set.intersection(*(set(metadata) for metadata in metadata_by_policy))
        assert common_slots
        for slot in common_slots:
            assert len({metadata[slot] for metadata in metadata_by_policy}) == 1


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


@pytest.mark.parametrize(
    ("seeds", "policies", "controls", "error_type"),
    [
        ((), POLICIES, {}, ValueError),
        ((3, 3), POLICIES, {}, ValueError),
        ((-1,), POLICIES, {}, ValueError),
        ((True,), POLICIES, {}, TypeError),
        ((3,), (), {}, ValueError),
        ((3,), ("uniform", "uniform"), {}, ValueError),
        ((3,), ("uniform", "unsupported"), {}, ValueError),
        ((3,), POLICIES, {"total_force_budget": 0}, ValueError),
        ((3,), POLICIES, {"action_force_budget": 0}, ValueError),
        ((3,), POLICIES, {"batch_size": 0}, ValueError),
        ((3,), POLICIES, {"max_workers": 3}, ValueError),
    ],
)
def test_run_comparison_preflights_all_inputs_without_creating_output_root(
    tmp_path: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    controls: dict[str, int],
    error_type: type[Exception],
):
    output_root = tmp_path / "must-not-exist"
    defaults = {
        "total_force_budget": 120,
        "action_force_budget": 30,
        "batch_size": 2,
        "max_workers": 2,
    }

    with pytest.raises(error_type):
        run_comparison(
            output_root=output_root,
            seeds=seeds,
            policies=policies,
            **(defaults | controls),
        )

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("seed_values", "controls"),
    [
        (("3", "3"), {}),
        (("-1",), {}),
        (("3",), {"total_force_budget": 0}),
        (("3",), {"max_workers": 3}),
    ],
)
def test_cli_invalid_inputs_do_not_create_output_or_derived_run_root(
    tmp_path: Path, seed_values: tuple[str, ...], controls: dict[str, int]
):
    output = tmp_path / "facts.json"
    defaults = {
        "total_force_budget": 120,
        "action_force_budget": 30,
        "batch_size": 2,
        "max_workers": 2,
    }
    effective = defaults | controls
    arguments = [
        "--output",
        str(output),
        "--seeds",
        *seed_values,
        "--total-force-budget",
        str(effective["total_force_budget"]),
        "--action-force-budget",
        str(effective["action_force_budget"]),
        "--batch-size",
        str(effective["batch_size"]),
        "--max-workers",
        str(effective["max_workers"]),
    ]

    with pytest.raises(ValueError):
        main(arguments)

    assert not output.exists()
    assert not (tmp_path / "facts-runs").exists()


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
