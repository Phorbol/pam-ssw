from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/20260803-two-operator-population-gate"


def load_runner():
    path = RUN / "replay_stage_a.py"
    spec = importlib.util.spec_from_file_location("_two_operator_stage_a", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_rejects_source_hash_drift(tmp_path):
    runner = load_runner()
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256"):
        runner.verify_source(source, "0" * 64)


def test_summarize_g_up0_records_no_new_force_evaluations():
    runner = load_runner()
    pairs = [
        {
            "system": "c60",
            "state_id": "plateau",
            "seed": 42,
            "causal_outcome": "RELAXED_ONLY_ESCAPE",
            "explicit_label": "RETURN_STARTER",
            "relaxed_label": "ESCAPED_CERTIFIED",
            "new_force_evaluations": 80,
            "explicit_purpose_counts": {
                "landing_true_quench": 79,
                "post_relax_validation": 1,
                "unattributed": 0,
            },
        }
    ]
    result = runner.summarize_pairs(pairs)
    assert result["new_force_evaluations"] == 0
    assert result["relaxed_only_escape_pairs"] == 1
    assert result["direct_quench_observed_force_evaluations"] == 80


def test_summarize_rejects_unclosed_historical_ledger():
    runner = load_runner()
    pairs = [
        {
            "causal_outcome": "RELAXED_ONLY_ESCAPE",
            "new_force_evaluations": 80,
            "explicit_purpose_counts": {
                "landing_true_quench": 78,
                "post_relax_validation": 1,
                "unattributed": 0,
            },
        }
    ]
    with pytest.raises(RuntimeError, match="does not close"):
        runner.summarize_pairs(pairs)
