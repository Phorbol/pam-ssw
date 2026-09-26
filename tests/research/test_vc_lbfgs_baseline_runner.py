import numpy as np
import pytest

from research.ga_ssw.vc_lbfgs_baseline_runner import temporary_lbfgs_baseline
from pamssw.standalone import vc_reference, cell_relax
from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart
from ase.build import bulk
from ase.calculators.emt import EMT


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


def test_bridge_records_first_common_passage_from_accepted_trace_without_extra_oracle_calls():
    calls = []
    records = []
    q0 = np.array([2., 0., 0.])
    evaluate = lambda q: (0.5 * float(q @ q), q)

    def counted(q):
        calls.append(q.copy())
        return evaluate(q)

    with temporary_lbfgs_baseline("ase", records):
        wrapped = vc_reference.safe_lbfgs(
            q0, counted, gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
            convergence_norm=lambda q, g: np.linalg.norm(g), gtol=1e-5,
            maxiter=20,
        )
    record = records[0]
    assert record["first_common_qualified_request"] is not None
    assert record["first_common_qualified_accepted_index"] is not None
    assert record["first_common_qualified_criterion"] <= 1e-5
    assert record["requests_after_first_common_qualified"] == (
        record["requests"] - record["first_common_qualified_request"]
    )
    assert len(calls) == record["requests"]

    from research.ga_ssw.lbfgs_baselines import ase_lbfgs_linesearch
    direct_calls = []
    direct = ase_lbfgs_linesearch(
        q0, lambda q: (direct_calls.append(q.copy()) or evaluate(q)),
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        convergence_norm=lambda q, g: np.linalg.norm(g), gtol=1e-5,
        maxiter=20,
    )
    assert len(direct_calls) == len(calls) == direct.requests == record["requests"]
    assert wrapped.status == direct.status
    assert np.array_equal(wrapped.q, direct.q)
    assert np.array_equal(wrapped.gradient, direct.gradient)
    assert wrapped.energy == direct.energy
    assert len(wrapped.trace) == len(direct.accepted_trace)
    for observed, expected in zip(wrapped.trace, direct.accepted_trace):
        assert np.array_equal(observed["q"], expected["q"])
        assert np.array_equal(observed["gradient"], expected["gradient"])
        assert observed["requests"] == expected["requests"]


def test_bridge_first_pass_is_none_when_no_accepted_point_passes_and_none_callback_uses_gradient_norm():
    records = []
    with temporary_lbfgs_baseline("scipy", records):
        vc_reference.safe_lbfgs(
            np.array([1., 0., 0.]), lambda q: (0.5 * float(q @ q), q),
            gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
            convergence_norm=None, gtol=1e-8, maxiter=0,
        )
    assert records[0]["first_common_qualified_request"] is None
    assert records[0]["first_common_qualified_accepted_index"] is None
    assert records[0]["first_common_qualified_criterion"] is None
    assert records[0]["requests_after_first_common_qualified"] is None


def test_observer_covers_true_cell_quench_emt_path():
    atoms = bulk("Cu", "fcc", a=3.65, cubic=True)
    atoms.positions[0] += [0.03, -0.02, 0.01]
    chart = SymmetricLogStrainChart(atoms, strain_length=3.6)
    q0 = chart.pack(atoms)
    surface = ASEStressSurface(EMT())

    records = []
    with temporary_lbfgs_baseline("ase", records, native_gradient_tol=0.005):
        true = cell_relax.relax_cell_coordinates(
            chart, q0, surface, pressure=0., fmax=.01, stress_tol=.001,
            max_step=.2, maxiter=50,
        )

    assert [row["source"] for row in records] == ["cell_relax"]
    assert true.requests == records[0]["requests"]
    assert records[0]["common_gtol"] == 1.
    assert records[0]["first_common_qualified_request"] is not None
    assert records[0]["first_common_qualified_criterion"] <= 1.
    assert records[0]["requests_after_first_common_qualified"] == (
        records[0]["requests"] - records[0]["first_common_qualified_request"]
    )


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
