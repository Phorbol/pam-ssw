"""Reject ambiguous cell-quench configurations before spending PES budget."""

import pytest
from pamssw import SSWConfig
from pamssw.exploration import SSWAttemptWorker


def test_cell_quench_is_explicit_and_fixed_is_the_default():
    assert SSWConfig().quench_cell_mode == "fixed"
    config = SSWConfig(quench_cell_mode="shape", quench_optimizer="ase-lbfgs")
    assert config.quench_stress_tol > 0


@pytest.mark.parametrize("kwargs", [
    {"quench_cell_mode": "invalid"},
    {"quench_cell_mode": "shape"},  # scipy Cartesian backend cannot relax cell
    {"quench_cell_mode": "shape", "quench_optimizer": "ase-fire", "quench_fallback_optimizer": "scipy-lbfgsb"},
    {"quench_stress_tol": float("nan")},
    {"quench_stress_tol": 0.},
    {"external_pressure_gpa": float("inf")},
    {"external_pressure_gpa": 1.},  # pressure cannot silently vanish in fixed mode
    {"dedup_cell_tol": -1.},
    {"quench_cell_mode": "slab_xy", "quench_optimizer": "ase-fire", "external_pressure_gpa": 1.},
])
def test_invalid_cell_quench_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        SSWConfig(**kwargs)


def test_posterior_cell_quench_is_explicitly_unsupported():
    config = SSWConfig(quench_cell_mode="shape", quench_optimizer="ase-lbfgs")
    with pytest.raises(ValueError, match="cell"):
        SSWAttemptWorker(lambda: None, config)
