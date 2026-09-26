"""CPU-only interface tests for the frozen VC optimizer panel; no MACE/PES jobs."""
import tempfile
import time
import unittest
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read

from research.ga_ssw.run_vc_frozen_optimizer_panel import (
    _atoms, _biased_value_gradient, _read_case, fresh_endpoint, run_stage,
)
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart


class FrozenVCPanelTest(unittest.TestCase):
    def test_aloh_chart_uses_original_run_input_and_saved_cells_roundtrip(self):
        plan_path = (Path(__file__).resolve().parents[1] /
                     "evidence/vc-mature-frozen-panel-20260926/plan.json")
        case = json.loads(plan_path.read_text())["cases"][0]
        _, chart, _, q_saved, terms, _, selected = _read_case(case)
        original_input = read(case["source_chart_input"])
        np.testing.assert_allclose(chart.reference.cell.array, original_input.cell.array,
                                   atol=1e-12, rtol=0)
        np.testing.assert_allclose(chart.unpack(q_saved).cell.array,
                                   np.asarray(selected["cell"]), atol=1e-8, rtol=0)
        source = json.loads(Path(case["source_result"]).read_text())["result"]
        initial_atoms = _atoms(source["initial"])
        reconstructed = chart.unpack(np.asarray(terms[0]["center"]))
        np.testing.assert_allclose(reconstructed.positions, initial_atoms.positions,
                                   atol=1e-9, rtol=0)
        np.testing.assert_allclose(reconstructed.cell.array, initial_atoms.cell.array,
                                   atol=1e-9, rtol=0)

    def setUp(self):
        # The original unstable simple-cubic one-atom fixture and its SciPy
        # cell-collapse failure are retained in preflight1-source/ and the
        # bounded diagnostic. This smoke test uses a physical fcc Cu cell.
        self.atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        self.chart = SymmetricLogStrainChart(self.atoms, strain_length=3.6)
        self.q_saved = self.chart.pack(self.atoms)
        direction = np.zeros(self.chart.ndof)
        direction[-1] = 1.0
        self.terms = [{"center": self.q_saved.tolist(), "direction": direction.tolist(),
                       "weight": 0.01, "width": 0.2}]
        self.q_start = self.q_saved + 0.2 * direction
        self.spec = {"pressure_eV_A3": 0.0, "gradient_tol_eV_A": 0.05,
                     "fmax_eV_A": 0.05, "stress_tol_eV_A3": 0.001,
                     "max_step_A": 0.2, "maxiter_per_stage": 2,
                     "request_cap_per_stage": 12, "lbfgs_history_pairs": 500}

    def calculator(self):
        return EMT()

    def test_projected_gaussian_gradient_finite_difference(self):
        rng = np.random.default_rng(19)
        q = rng.normal(size=9)
        center = rng.normal(size=9)
        direction = rng.normal(size=9)
        direction /= np.linalg.norm(direction)
        terms = [{"center": center, "direction": direction,
                  "weight": 0.17, "width": 0.31}]
        base_e, base_g = 0.4, rng.normal(size=9)
        value, gradient = _biased_value_gradient(q, base_e, base_g, terms)
        eps = 1e-6
        numeric = np.empty(9)
        for i in range(9):
            delta = np.zeros(9); delta[i] = eps
            vp = _biased_value_gradient(q + delta, base_e + base_g @ delta,
                                        base_g, terms)[0]
            vm = _biased_value_gradient(q - delta, base_e - base_g @ delta,
                                        base_g, terms)[0]
            numeric[i] = (vp-vm)/(2*eps)
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-7)

    def test_three_backends_two_shared_starts_and_paid_cache_telemetry(self):
        methods = ("safe-lbfgs-total", "ase-lbfgs-linesearch", "scipy-lbfgsb")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for method in methods:
                for phase, start, terms in (("biased", self.q_start, self.terms),
                                            ("unbiased", self.q_saved, [])):
                    deadline = time.monotonic() + 60
                    result = run_stage(method, phase, start, self.spec, self.chart,
                        terms, self.calculator, root / f"{method}-{phase}", deadline)
                    self.assertLessEqual(result["search_requests"], self.spec["request_cap_per_stage"])
                    request_ids = {row["request"] for row in result["accepted_iterates"]}
                    first = result["first_common_qualified_request"]
                    self.assertTrue(first is None or first in request_ids)
                    ledger_ids = {int(__import__("json").loads(line)["request"])
                                  for line in Path(result["request_ledger"]).read_text().splitlines()
                                  if __import__("json").loads(line).get("request", 0) > 0}
                    self.assertEqual(len(ledger_ids), result["search_requests"])
                    fresh = fresh_endpoint(np.asarray(result["terminal_q"]), self.spec,
                                           self.chart, terms, self.calculator)
                    self.assertEqual(fresh["requests"], 1)
                    results.append(result)
            self.assertEqual(len(results), 6)

    def test_low_request_cap_keeps_paid_initial_cost(self):
        spec = dict(self.spec, request_cap_per_stage=1, maxiter_per_stage=10)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stage("safe-lbfgs-total", "biased", self.q_start, spec,
                self.chart, self.terms, self.calculator, Path(tmp) / "low-cap",
                time.monotonic() + 60)
        self.assertEqual(result["search_requests"], 1)
        self.assertIn(result["status"], {"request_limit", "converged"})


if __name__ == "__main__":
    unittest.main()
