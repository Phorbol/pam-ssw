from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from ase.io import write
import numpy as np
import pytest

from pamssw.io import state_to_atoms
from pamssw.state import State


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT / "runs" / "20260731-uphill-relax-first-passage-gate"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
RUNNER_PATH = RUN_ROOT / "run_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol():
    return _load(PROTOCOL_PATH, "_uphill_relax_first_passage_protocol_test")


def _runner():
    return _load(RUNNER_PATH, "_uphill_relax_first_passage_runner_test")


def test_discovery_is_only_the_four_repeated_causal_trajectories():
    protocol = _protocol()

    assert protocol.DISCOVERY_KEYS == (
        ("plateau_accepted", 42, "D0_exact_anchor", 1),
        ("plateau_accepted", 43, "D0_exact_anchor", 1),
        ("plateau_accepted", 43, "K4_discrete", 2),
        ("plateau_accepted", 44, "K4_discrete", 2),
    )


def test_holdout_contains_the_other_four_mechanism_informative_pairs():
    protocol = _protocol()

    assert len(protocol.HOLDOUT_KEYS) == 4
    assert set(protocol.DISCOVERY_KEYS).isdisjoint(protocol.HOLDOUT_KEYS)


def _frames(*classes: str) -> list[dict[str, object]]:
    mapping = {
        "RETURN": ("RETURN_STARTER", "NOT_APPLICABLE"),
        "FINAL": ("ESCAPED_CERTIFIED", "SAME_LANDING"),
        "OTHER": ("ESCAPED_CERTIFIED", "DIFFERENT_LANDING"),
        "INVALID": ("INVALID_GEOMETRY", "NOT_COMPARABLE"),
    }
    return [
        {
            "frame_index": index,
            "label": mapping[value][0],
            "landing_relation_to_final": mapping[value][1],
        }
        for index, value in enumerate(classes)
    ]


def test_stable_final_basin_step_uses_the_complete_final_suffix():
    protocol = _protocol()
    rows = _frames("RETURN", "FINAL", "OTHER", "FINAL", "FINAL")

    result = protocol.summarize_trajectory(rows)

    assert result == {
        "frame_count": 5,
        "final_frame_index": 4,
        "first_escape_step": 1,
        "stable_escape_step": 1,
        "stable_final_basin_step": 3,
        "unlearnable_frame_count": 0,
    }


def test_unlearnable_frame_breaks_the_stable_suffix():
    protocol = _protocol()
    rows = _frames("RETURN", "FINAL", "INVALID", "FINAL", "FINAL")

    result = protocol.summarize_trajectory(rows)

    assert result["first_escape_step"] == 1
    assert result["stable_escape_step"] == 3
    assert result["stable_final_basin_step"] == 3
    assert result["unlearnable_frame_count"] == 1


def test_cutoff_is_the_maximum_stable_final_arrival_without_weights():
    protocol = _protocol()
    summaries = [
        {"stable_final_basin_step": value}
        for value in (7, 11, 4, 9)
    ]

    assert protocol.derive_cutoff(summaries) == 11


def test_cutoff_requires_all_four_discovery_trajectories():
    protocol = _protocol()

    with pytest.raises(ValueError, match="four discovery"):
        protocol.derive_cutoff(
            [{"stable_final_basin_step": value} for value in (2, 3, 4)]
        )
    with pytest.raises(ValueError, match="stable final-basin"):
        protocol.derive_cutoff(
            [
                {"stable_final_basin_step": 2},
                {"stable_final_basin_step": 3},
                {"stable_final_basin_step": None},
                {"stable_final_basin_step": 4},
            ]
        )


def _holdout_row(
    key,
    *,
    has_headroom: bool = True,
    relation: str = "SAME_LANDING",
    label: str = "ESCAPED_CERTIFIED",
) -> dict[str, object]:
    state_id, seed, arm, horizon = key
    return {
        "state_id": state_id,
        "seed": seed,
        "arm": arm,
        "horizon": horizon,
        "status": "completed",
        "has_headroom": has_headroom,
        "cutoff_label": label,
        "cutoff_relation_to_final": relation,
    }


def test_holdout_opens_only_when_every_untouched_path_reproduces_final_basin():
    protocol = _protocol()
    passing = [_holdout_row(key) for key in protocol.HOLDOUT_KEYS]
    no_headroom = list(passing)
    no_headroom[0] = _holdout_row(
        protocol.HOLDOUT_KEYS[0],
        has_headroom=False,
    )
    changed = list(passing)
    changed[1] = _holdout_row(
        protocol.HOLDOUT_KEYS[1],
        relation="DIFFERENT_LANDING",
    )

    assert protocol.decide_holdout(passing)["decision"] == (
        "OPEN_FIXED_CUTOFF_FULL_ACTION_GATE"
    )
    assert protocol.decide_holdout(no_headroom)["decision"] == (
        "RETAIN_CURRENT_LENGTH_NO_PROMOTION"
    )
    assert protocol.decide_holdout(changed)["decision"] == (
        "RETAIN_CURRENT_LENGTH_NO_PROMOTION"
    )


def _state(offset: float) -> State:
    return State(
        numbers=np.asarray([6, 6]),
        positions=np.asarray(
            [[offset, 0.0, 0.0], [1.4 + offset, 0.0, 0.0]],
        ),
        cell=None,
        pbc=(False, False, False),
        fixed_mask=np.asarray([True, False]),
    )


def test_extract_frame_preserves_requested_accepted_step(tmp_path):
    runner = _runner()
    states = [_state(value) for value in (0.0, 0.1, 0.2)]
    path = tmp_path / "trace.xyz"
    write(path, [state_to_atoms(state) for state in states])

    extracted = runner.extract_frame(path, 1, states[0])

    np.testing.assert_allclose(extracted.positions, states[1].positions)
    np.testing.assert_array_equal(extracted.fixed_mask, [True, False])


def test_reused_frame_zero_costs_no_new_force_evaluations():
    runner = _runner()
    source = {
        "state_id": "plateau_accepted",
        "seed": 42,
        "arm": "D0_exact_anchor",
        "horizon": 1,
        "explicit_label": "RETURN_STARTER",
        "explicit_landing_energy_eV": -10.0,
        "explicit_landing_path": "explicit.xyz",
        "explicit_landing_sha256": "a" * 64,
        "landing_relation": "NOT_APPLICABLE",
        "relaxed_label": "ESCAPED_CERTIFIED",
        "relaxed_landing_energy_eV": -11.0,
        "relaxed_landing_path": "relaxed.xyz",
        "relaxed_landing_sha256": "b" * 64,
    }

    row = runner.reused_endpoint_row(
        source,
        frame_index=0,
        final_frame_index=80,
    )

    assert row["label"] == "RETURN_STARTER"
    assert row["new_force_evaluations"] == 0
    assert row["evidence_origin"] == "reused_g_up0"


def test_intermediate_quench_creates_its_case_directory(tmp_path):
    runner = _runner()
    starter = _state(0.0)
    trajectory = tmp_path / "trace.xyz"
    write(
        trajectory,
        [state_to_atoms(_state(value)) for value in (0.0, 0.1, 0.2)],
    )
    case_dir = tmp_path / "not-yet-created" / "case"
    key = ("plateau_accepted", 42, "D0_exact_anchor")

    def quench_checkpoint(**kwargs):
        assert kwargs["case_dir"].parent == case_dir
        return {
            "label": "ESCAPED_CERTIFIED",
            "landing_energy_eV": -11.0,
            "landing_path": "landing.xyz",
            "landing_sha256": "a" * 64,
            "force_evaluations": 3,
            "purpose_counts": {**runner._zero_counts(), "landing_true_quench": 3},
            "wall_time_s": 0.2,
        }

    runtime = {
        "states": {"plateau_accepted": (starter, {"state_sha256": "b" * 64})},
        "action_runner": SimpleNamespace(_base_config=lambda *args: {}),
        "config_gate": object(),
        "first_passage": SimpleNamespace(_quench_checkpoint=quench_checkpoint),
        "calculator": object(),
        "base_runner": object(),
        "gup0": SimpleNamespace(
            _landing_relation=lambda **kwargs: ("SAME_LANDING", {})
        ),
    }
    pair = {
        "state_id": key[0],
        "seed": key[1],
        "arm": key[2],
        "horizon": 1,
        "source_optimizer_trajectory_path": str(trajectory),
        "relaxed_label": "ESCAPED_CERTIFIED",
        "relaxed_landing_energy_eV": -11.0,
        "relaxed_landing_path": "final.xyz",
        "relaxed_landing_sha256": "c" * 64,
    }
    context = {
        "first_passage_cases": {
            key: {"starter_energy_eV": -10.0},
        }
    }

    row = runner._quench_intermediate_frame(
        pair=pair,
        frame_index=1,
        final_frame_index=2,
        case_dir=case_dir,
        context=context,
        runtime=runtime,
        remaining_budget=10,
    )

    assert case_dir.is_dir()
    assert row["new_force_evaluations"] == 3
