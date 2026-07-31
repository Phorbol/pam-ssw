from __future__ import annotations

import importlib.util
from hashlib import sha256
from pathlib import Path
import sys

from ase.io import write
import numpy as np
import pytest

from pamssw.io import state_to_atoms
from pamssw.state import State


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260731-uphill-relax-counterfactual-gate"
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
    return _load(PROTOCOL_PATH, "_uphill_relax_counterfactual_protocol_test")


def _runner():
    return _load(RUNNER_PATH, "_uphill_relax_counterfactual_runner_test")


def _source_cases() -> list[dict[str, object]]:
    cases = []
    for state_id in ("intermediate_accepted", "plateau_accepted"):
        for seed in (42, 43, 44):
            for arm in ("D0_exact_anchor", "K4_discrete"):
                horizons = [1, 2, 4]
                if arm == "D0_exact_anchor" and (
                    (state_id, seed)
                    in {
                        ("intermediate_accepted", 43),
                        ("plateau_accepted", 42),
                    }
                ):
                    horizons = [1, 2]
                cases.append(
                    {
                        "system": "c60",
                        "state_id": state_id,
                        "seed": seed,
                        "arm": arm,
                        "checkpoints": [
                            {
                                "horizon": horizon,
                                "raw_optimizer_checkpoint_path": (
                                    f"trace-{state_id}-{seed}-{arm}-{horizon}.xyz"
                                ),
                                "raw_optimizer_checkpoint_sha256": (
                                    f"hash-{state_id}-{seed}-{arm}-{horizon}"
                                ),
                                "label": "ESCAPED_CERTIFIED",
                            }
                            for horizon in horizons
                        ],
                    }
                )
    cases.append(
        {
            "system": "pdo",
            "state_id": "intermediate_accepted",
            "seed": 42,
            "arm": "D0_exact_anchor",
            "checkpoints": [],
        }
    )
    return cases


def test_pair_specs_use_only_c60_horizons_1_2_4_with_recorded_relaxation():
    protocol = _protocol()

    specs = protocol.pair_specs(_source_cases())

    assert len(specs) == 34
    assert {spec["system"] for spec in specs} == {"c60"}
    assert {spec["horizon"] for spec in specs} == {1, 2, 4}
    assert all(spec["raw_optimizer_checkpoint_path"] for spec in specs)


@pytest.mark.parametrize(
    ("explicit_label", "relaxed_label", "landing_relation", "expected"),
    [
        (
            "RETURN_STARTER",
            "RETURN_STARTER",
            "NOT_APPLICABLE",
            "BOTH_RETURN_STARTER",
        ),
        (
            "ESCAPED_CERTIFIED",
            "ESCAPED_CERTIFIED",
            "SAME_LANDING",
            "SAME_ESCAPED_LANDING",
        ),
        (
            "ESCAPED_CERTIFIED",
            "ESCAPED_CERTIFIED",
            "DIFFERENT_LANDING",
            "DIFFERENT_ESCAPED_LANDINGS",
        ),
        (
            "RETURN_STARTER",
            "ESCAPED_CERTIFIED",
            "NOT_APPLICABLE",
            "RELAXED_ONLY_ESCAPE",
        ),
        (
            "ESCAPED_CERTIFIED",
            "RETURN_STARTER",
            "NOT_APPLICABLE",
            "EXPLICIT_ONLY_ESCAPE",
        ),
        (
            "INVALID_GEOMETRY",
            "ESCAPED_CERTIFIED",
            "NOT_COMPARABLE",
            "UNLEARNABLE",
        ),
    ],
)
def test_causal_outcome_uses_only_certified_basin_labels(
    explicit_label,
    relaxed_label,
    landing_relation,
    expected,
):
    protocol = _protocol()

    assert protocol.causal_outcome(
        explicit_label=explicit_label,
        relaxed_label=relaxed_label,
        landing_relation=landing_relation,
    ) == expected


def _decision_row(seed: int, outcome: str) -> dict[str, object]:
    return {
        "system": "c60",
        "state_id": "intermediate_accepted",
        "seed": seed,
        "arm": "K4_discrete",
        "horizon": 2,
        "causal_outcome": outcome,
    }


def test_repeated_relaxed_only_escape_retains_relaxation():
    protocol = _protocol()
    rows = [
        _decision_row(42, "RELAXED_ONLY_ESCAPE"),
        _decision_row(43, "RELAXED_ONLY_ESCAPE"),
        _decision_row(44, "SAME_ESCAPED_LANDING"),
    ]

    result = protocol.decide(rows)

    assert result["decision"] == "RETAIN_RELAXATION_CAUSAL_SIGNAL"
    assert result["repeated_relaxed_only_contexts"] == [
        {
            "system": "c60",
            "state_id": "intermediate_accepted",
            "arm": "K4_discrete",
            "horizon": 2,
            "seed_count": 2,
            "seeds": [42, 43],
        }
    ]


def test_exact_reproduction_is_the_only_redundancy_signal():
    protocol = _protocol()
    exact = [
        _decision_row(42, "BOTH_RETURN_STARTER"),
        _decision_row(43, "SAME_ESCAPED_LANDING"),
        _decision_row(44, "SAME_ESCAPED_LANDING"),
    ]
    mixed = exact[:-1] + [
        _decision_row(44, "DIFFERENT_ESCAPED_LANDINGS")
    ]

    assert protocol.decide(exact)["decision"] == "EXACT_REDUNDANCY_SIGNAL"
    assert protocol.decide(mixed)["decision"] == "MIXED_NO_DEFAULT_CHANGE"


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


def test_extract_explicit_state_returns_first_optimizer_frame(tmp_path):
    runner = _runner()
    first = _state(0.1)
    second = _state(0.2)
    path = tmp_path / "trace.xyz"
    write(path, [state_to_atoms(first), state_to_atoms(second)])

    replayed = runner.extract_explicit_state(path, _state(0.0))

    np.testing.assert_allclose(replayed.positions, first.positions)
    np.testing.assert_array_equal(replayed.fixed_mask, [True, False])


def test_source_checkpoint_hash_drift_is_rejected(tmp_path):
    runner = _runner()
    path = tmp_path / "trace.xyz"
    path.write_bytes(b"immutable source")
    expected = sha256(path.read_bytes()).hexdigest()

    runner.verify_source_file(path, expected)
    with pytest.raises(RuntimeError, match="SHA256"):
        runner.verify_source_file(path, "0" * 64)
