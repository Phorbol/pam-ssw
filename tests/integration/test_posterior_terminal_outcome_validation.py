from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-posterior-terminal-outcome-validation"
    / "run_validation.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "posterior_terminal_outcome_validation",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load validation script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_writes_only_closed_replayable_evidence(tmp_path: Path) -> None:
    validator = _load_validator()
    evidence = validator.run_validation(tmp_path, source_commit="0" * 40)

    persisted = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    conclusion = (tmp_path / "conclusion.md").read_text(encoding="utf-8")

    assert persisted == evidence
    assert evidence["schema_version"] == 1
    assert evidence["git_commit"] == "0" * 40
    assert evidence["real_campaign"]["configuration"]["proposal_pool_size"] == 1
    assert evidence["real_campaign"]["configuration"]["batch_size"] > (
        evidence["real_campaign"]["configuration"]["max_workers"]
    ) > 1
    assert all(evidence["invariants"].values())
    assert [
        row["posterior_observed"] for row in evidence["terminal_matrix"]["rows"]
    ] == [True, True, True, True, False, False]
    assert evidence["terminal_matrix"]["live_posterior"] == (
        evidence["terminal_matrix"]["replayed_posterior"]
    )
    assert "does not compare starter-policy performance" in conclusion
