import numpy as np
import pytest

from research.ga_ssw.vc_lbfgs_baseline_runner import temporary_lbfgs_baseline
from pamssw.standalone import vc_reference, cell_relax


def test_baseline_context_restores_production_call_sites_and_has_no_secant_fiction():
    old_vc, old_cell = vc_reference.safe_lbfgs, cell_relax.safe_lbfgs
    records = []
    with temporary_lbfgs_baseline("scipy", records):
        assert vc_reference.safe_lbfgs is not old_vc
        assert cell_relax.safe_lbfgs is not old_cell
    assert vc_reference.safe_lbfgs is old_vc
    assert cell_relax.safe_lbfgs is old_cell

    with pytest.raises(ValueError):
        with temporary_lbfgs_baseline("unknown"):
            pass


def test_baseline_bridge_records_native_and_common_fields_without_secants():
    records = []
    with temporary_lbfgs_baseline("ase", records):
        result = vc_reference.safe_lbfgs(
            np.array([1.0, 0.0, 0.0]), lambda q: (0.5 * float(q @ q), q),
            gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
            convergence_norm=lambda q, g: np.linalg.norm(g), gtol=1e-8,
            max_step=0.2, maxiter=2,
        )
    assert records[0]["max_step"] == 0.2
    assert "certificate" in records[0] and "native_success" in records[0]
    assert not hasattr(result, "accepted_secants")
    assert result.rejected_trials >= 0


@pytest.mark.parametrize("kind", ["scipy", "ase"])
def test_biased_vc_norm_conversion_is_sufficient_bound_and_not_true_quench_conversion(kind):
    records = []
    with temporary_lbfgs_baseline(kind, records, native_gradient_tol=0.005):
        vc_reference.safe_lbfgs(
            np.array([1.0, 0.0, 0.0]), lambda q: (0.5 * float(q @ q), q),
            gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
            gtol=0.006, maxiter=0,
        )
    conversion = records[0]["norm_conversion"]
    assert conversion["sufficient"] is True and conversion["necessary"] is False
    if kind == "scipy":
        assert records[0]["metadata"]["scipy_options"]["gtol"] == pytest.approx(0.006 / np.sqrt(6))
    else:
        assert records[0]["metadata"]["ase_parameters"]["fmax"] == pytest.approx(0.006 / np.sqrt(2))
