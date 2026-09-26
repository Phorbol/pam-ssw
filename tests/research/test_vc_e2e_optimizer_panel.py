"""CPU-only integration contract for the complete VC optimizer worker."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from pamssw.standalone import vc_reference, cell_relax
from pamssw.standalone.vc_reference import VCSSWConfig
from research.ga_ssw.run_vc_e2e_optimizer_panel import run_arm


def config():
    return VCSSWConfig(strain_length=3.6, width=0.2, rotation_bias=0.5,
        max_gaussians=1, gradient_tol=0.05, fmax=0.05, stress_tol=0.001,
        max_step=0.2, relax_steps=12, rotation_hvp=9, rotation_tol=0.02,
        lbfgs_memory=500, bias_release="strict")


class VCE2EOptimizerPanelTest(unittest.TestCase):
    def atoms(self):
        return bulk("Cu", "fcc", a=3.615, cubic=True)

    def calculator(self):
        return EMT()

    def test_three_methods_preserve_context_ledger_cost_and_record_backed_fresh(self):
        methods = ("safe_total", "ase", "scipy")
        original_vc = vc_reference.safe_lbfgs
        original_cell = cell_relax.safe_lbfgs
        with tempfile.TemporaryDirectory() as tmp:
            for method in methods:
                out = Path(tmp) / method
                result = run_arm(self.atoms(), self.calculator, config(),
                    method=method, seed=7, steps=1, request_cap=240,
                    search_seconds=60, output_dir=out,
                    case_name="Cu4-EMT-test")
                self.assertIs(vc_reference.safe_lbfgs, original_vc)
                self.assertIs(cell_relax.safe_lbfgs, original_cell)
                self.assertLessEqual(result["search_requests"], 240)
                with (out / "search-ledger.jsonl").open() as stream:
                    ledger = [json.loads(line) for line in stream if line.strip()]
                starts = [row for row in ledger if row["event"] == "attempt_started"]
                completions = [row for row in ledger if row["event"] in
                               ("attempt_completed", "attempt_error", "budget_censor")]
                self.assertEqual(len(starts), len(completions))
                self.assertEqual(sum(row.get("charged", False) for row in completions),
                                 result["search_requests"])
                search_result = json.loads((out / "search-result.json").read_text())
                self.assertTrue(any(row.get("index") == 0 for row in
                                    search_result["result"]["records"]),
                    "full-path smoke must enter outer attempt 0; initial-only failure is insufficient")
                expected = [("initial", -1)] if search_result["result"]["initial"] is not None else []
                expected += [("landing", row["index"]) for row in search_result["result"]["records"]
                             if isinstance(row, dict) and row.get("landing") is not None]
                fresh = json.loads((out / "fresh-checks.json").read_text())
                self.assertEqual([(row["source"], row["record_index"])
                                  for row in fresh["checks"]], expected)
                self.assertLessEqual(fresh["requests"], 4)
                fresh_ledger = [json.loads(line) for line in
                    (out / "fresh-ledger.jsonl").read_text().splitlines()]
                self.assertEqual(sum(row["event"] == "fresh_started" for row in fresh_ledger),
                                 len(expected))
                self.assertEqual(sum(row["event"] in ("fresh_completed", "fresh_error")
                                     for row in fresh_ledger), len(expected))
                self.assertEqual(sum(bool(row.get("charged", False)) and
                                     row["event"] in ("fresh_completed", "fresh_error")
                                     for row in fresh_ledger), fresh["requests"])
                for row in fresh["checks"]:
                    if row.get("status") == "checked":
                        certificate = row["physical_certificate"]
                        self.assertEqual(certificate["force_pass"],
                                         certificate["fmax_eV_A"] <= config().fmax)
                        self.assertEqual(certificate["stress_pass"],
                                         certificate["stress_residual_max_eV_A3"] <= config().stress_tol)

    def test_global_cap_is_reported_as_censor_with_paid_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "capped"
            result = run_arm(self.atoms(), self.calculator, config(),
                method="safe_total", seed=7, steps=1, request_cap=1,
                search_seconds=60, output_dir=out, case_name="Cu4-EMT-cap")
            self.assertTrue(result["budget_censor"])
            self.assertEqual(result["search_requests"], 1)
            ledger = [json.loads(line) for line in (out / "search-ledger.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["event"] == "budget_censor" for row in ledger))
            self.assertTrue((out / "search-result.json").is_file())


if __name__ == "__main__":
    unittest.main()
