from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = (
    ROOT / "runs/20260801-fixed-step-target-gate/analyze_results.py"
)


def _analyzer():
    name = "_fixed_step_target_analysis_test"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_first_best_force_evaluations_uses_first_attainment():
    analyzer = _analyzer()
    rows = [
        {"force_evaluations": 100, "best_energy": -2.0},
        {"force_evaluations": 300, "best_energy": -3.0},
        {"force_evaluations": 500, "best_energy": -3.0},
    ]
    assert analyzer.first_best_force_evaluations(-3.0, rows) == 300
    with pytest.raises(ValueError, match="never attained"):
        analyzer.first_best_force_evaluations(-4.0, rows)


def test_combine_cases_requires_exact_three_seed_matrix():
    analyzer = _analyzer()
    rows = [
        {
            "system": system,
            "seed": seed,
            "target_mode": mode,
            "gain_auc_eV": 1.0,
        }
        for system in analyzer.SYSTEMS
        for seed in (46, 47, 48)
        for mode in analyzer.TARGET_MODES
    ]
    combined = analyzer.combine_cases(rows[:6], rows[6:])
    assert len(combined) == 18
    with pytest.raises(ValueError, match="required matrix"):
        analyzer.combine_cases(rows[:6], rows[6:-1])


def test_target_distribution_is_finite_and_exact():
    analyzer = _analyzer()
    summary = analyzer.target_distribution([0.8, 0.2, 1.4])
    assert summary == {
        "count": 3,
        "mean_eV": pytest.approx(0.8),
        "median_eV": pytest.approx(0.8),
        "min_eV": pytest.approx(0.2),
        "max_eV": pytest.approx(1.4),
    }
    with pytest.raises(ValueError, match="non-empty"):
        analyzer.target_distribution([])
