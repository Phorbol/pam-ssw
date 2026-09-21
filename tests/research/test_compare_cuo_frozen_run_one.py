import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import research.ga_ssw.compare_cuo_frozen_quenches as diagnosis
from pamssw.standalone.vc_reference import VCSSWConfig
from pamssw.standalone.generalized_numerics import GeneralizedRelaxResult


class MockCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = np.asarray(atoms.positions)
        self.results["energy"] = float(np.square(x[:, 0]).sum())
        self.results["forces"] = np.column_stack((-2.0 * x[:, 0],
                                                    np.zeros((len(x), 2))))
        self.results["stress"] = np.zeros((3, 3))


def test_run_one_uses_frozen_bias_and_final_q_with_329_budget(tmp_path, monkeypatch):
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=3.0 * np.eye(3), pbc=True)
    q0 = np.zeros(9)
    direction = np.zeros(9)
    direction[0] = 1.0
    record = {
        "index": 1,
        "chart_reference": {"numbers": [29], "positions": [[0.0, 0.0, 0.0]],
                             "cell": (3.0 * np.eye(3)).tolist(), "pbc": [True, True, True]},
        "frozen_gaussians": [{"center": q0.tolist(), "direction": direction.tolist(),
                              "weight": 2.0, "width": 1.0}],
        "climb": [{"q": (0.2 * direction).tolist(),
                   "relaxation": {"gradient_norm": 1.0}}],
    }
    joint = VCSSWConfig(strain_length=3.0, width=.6, rotation_bias=100.,
                        gradient_tol=.001, fmax=.001, stress_tol=.0001,
                        max_step=.2, relax_steps=300, rotation_hvp=100,
                        rotation_tol=.02, fd_step=1e-4, max_gaussians=14,
                        temperature_K=300., forward_force=.1, lbfgs_memory=10)
    calls = []

    def fake_safe(q_start, evaluate, **kwargs):
        q = np.asarray(q_start, dtype=float).copy()
        # Leave exactly one request for _run_one's final fresh endpoint check.
        assert kwargs["max_requests"] == 329
        for i in range(329):
            q = q_start + (i + 1) * 1e-8 * direction
            energy, gradient = evaluate(q)
            calls.append((float(energy), np.asarray(gradient).copy()))
        return GeneralizedRelaxResult(q=q, status="maxiter", steps=300, requests=329,
                                      trace=tuple(), error=None, energy=float(energy),
                                      gradient=np.asarray(gradient), rejected_trials=0,
                                      accepted_secants=0, rejected_secants=0)

    monkeypatch.setattr(diagnosis, "safe_lbfgs", fake_safe)
    diagnosis._run_one(
        {"seed": 7, "baseline_attempted_requests": 315}, record, joint,
        MockCalculator, tmp_path, 10, float("inf"))

    result = __import__("json").loads((tmp_path / "history10" / "result.json").read_text())
    assert result["requests"] == 330
    assert len(calls) == 329
    # At q_start=(width*direction), the positive Gaussian adds 2*exp(-1/2).
    assert calls[0][0] > 0.0
    final = result["final_fresh"]
    assert final["objective"] > final["energy"]
    assert result["censored"] is False
