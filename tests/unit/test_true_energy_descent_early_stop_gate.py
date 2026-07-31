from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT / "runs" / "20260801-true-energy-descent-early-stop-gate"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
MANIFEST_PATH = RUN_ROOT / "manifest.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol():
    return _load(PROTOCOL_PATH, "_true_energy_descent_protocol_test")


def test_accepted_steps_exclude_rejected_attempts():
    protocol = _protocol()
    case = {"reached_macro_steps": 2, "attempted_macro_steps": 3}

    assert protocol.accepted_step_indices(case) == (1, 2)


def test_accepted_steps_reject_inconsistent_counts():
    protocol = _protocol()

    with pytest.raises(ValueError, match="reached/attempted"):
        protocol.accepted_step_indices(
            {"reached_macro_steps": 3, "attempted_macro_steps": 2}
        )


def test_first_crossing_is_strict_and_does_not_look_ahead():
    protocol = _protocol()
    rows = [
        {"step": 1, "checkpoint_delta_eV": 0.2},
        {"step": 2, "checkpoint_delta_eV": -0.001},
        {"step": 3, "checkpoint_delta_eV": -0.4},
    ]

    assert protocol.first_descent_crossing(
        rows,
        tolerance=0.001,
    ) == rows[2]


def test_first_crossing_requires_consecutive_accepted_steps():
    protocol = _protocol()
    rows = [
        {"step": 1, "checkpoint_delta_eV": 0.2},
        {"step": 3, "checkpoint_delta_eV": -0.4},
    ]

    with pytest.raises(ValueError, match="consecutive"):
        protocol.first_descent_crossing(rows, tolerance=0.001)


def test_outcome_keeps_overshoot_and_deeper_terminal_separate():
    protocol = _protocol()

    assert protocol.classify_tradeoff(-7.0, 0.2, 0.001) == (
        "AVOIDED_OVERSHOOT"
    )
    assert protocol.classify_tradeoff(-7.0, -9.0, 0.001) == (
        "FORGONE_DEEPER_TERMINAL"
    )
    assert protocol.classify_tradeoff(-7.0, -7.0, 0.001) == (
        "ENERGY_EQUIVALENT"
    )
    assert protocol.classify_tradeoff(None, -7.0, 0.001) == "UNLEARNABLE"


def test_manifest_uses_only_reached_steps(tmp_path):
    protocol = _protocol()
    case_dir = tmp_path / "case"
    checkpoint_dir = case_dir / "macro_checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step in (1, 2, 3):
        (checkpoint_dir / f"step{step:03d}_checkpoint.xyz").write_text(
            f"frame {step}\n",
            encoding="utf-8",
        )
    source_case = {
        "system": "c60",
        "state_id": "plateau_accepted",
        "seed": 42,
        "arm": "D0_exact_anchor",
        "reached_macro_steps": 2,
        "attempted_macro_steps": 3,
    }

    rows = protocol.project_manifest_case(
        source_case,
        case_dir,
        root=tmp_path,
    )

    assert [row["step"] for row in rows] == [1, 2]
    assert [row["checkpoint_path"] for row in rows] == [
        "case/macro_checkpoints/step001_checkpoint.xyz",
        "case/macro_checkpoints/step002_checkpoint.xyz",
    ]
    assert all(len(row["checkpoint_sha256"]) == 64 for row in rows)


def test_frozen_manifest_closes_exact_cohort():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["case_count"] == 24
    assert manifest["accepted_endpoint_count"] == 82
    assert manifest["attempted_endpoint_count"] == 93
    assert len(manifest["cases"]) == 24
    assert sum(len(case["endpoints"]) for case in manifest["cases"]) == 82
