import json
import sys
import types
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import research.ga_ssw.qualify_cuo_vc_input as qualify
from research.ga_ssw.qualify_cuo_vc_input import BudgetSurface, _evaluate
from pamssw.standalone.vc_geometry import VCEvaluation
from pamssw.standalone.cell_relax import CellQuenchResult


class QuadraticMock(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = np.asarray(atoms.positions)
        self.results["energy"] = float(np.square(x[:, 0]).sum())
        self.results["forces"] = np.column_stack((-2.0 * x[:, 0],
                                                    np.zeros((len(x), 2))))
        self.results["stress"] = np.zeros((3, 3))


def _cell_atom(x):
    return Atoms("Cu", positions=[[x, 0.0, 0.0]], cell=3.0 * np.eye(3), pbc=True)


def test_main_fresh_evaluates_quench_endpoint(tmp_path, monkeypatch):
    source = _cell_atom(0.0)
    source_data = {"symbols": source.get_chemical_symbols(),
                   "positions": source.positions.tolist(),
                   "cell": source.cell.array.tolist(), "pbc": source.pbc.tolist()}
    (tmp_path / "source-input.json").write_text(json.dumps(source_data))
    model = tmp_path / "model.pt"
    model.write_bytes(b"mock model")
    plan = {"model": str(model), "caps": {"max_EFS": 1500, "seconds": 600}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(_index):
            return "mock-cuda"

        @staticmethod
        def empty_cache():
            return None

    fake_torch = types.SimpleNamespace(
        __version__="mock-torch", cuda=FakeCuda(),
        set_num_threads=lambda _n: None,
        set_num_interop_threads=lambda _n: None,
        get_num_threads=lambda: 1,
        get_num_interop_threads=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeMACECalculator(QuadraticMock):
        def __init__(self, **_kwargs):
            super().__init__()

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            self.results["energy"] = 0.0
            self.results["forces"] = np.zeros((len(atoms), 3))
            self.results["stress"] = np.zeros((3, 3))

    fake_mace = types.ModuleType("mace")
    fake_calculators = types.ModuleType("mace.calculators")
    fake_calculators.MACECalculator = FakeMACECalculator
    fake_mace.calculators = fake_calculators
    monkeypatch.setitem(sys.modules, "mace", fake_mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", fake_calculators)

    endpoint_atoms = source.copy()
    endpoint_atoms.positions[:, 0] = 0.7
    endpoint = VCEvaluation(
        objective=0.0, gradient=np.zeros(9), atoms=endpoint_atoms,
        energy=0.0, forces=np.zeros((1, 3)), stress=np.zeros((3, 3)), volume=27.0)
    optimizer = SimpleNamespace(converged=True, status="converged", error=None)
    fake_quench = CellQuenchResult(
        evaluation=endpoint, optimizer=optimizer,
        certificate={"certified": True}, requests=1)
    monkeypatch.setattr(qualify, "cell_quench", lambda *args, **kwargs: fake_quench)
    monkeypatch.setattr(qualify, "null_space",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop before Hessian")))

    assert qualify.main(["--root", str(tmp_path)]) == 0

    rows = [json.loads(line) for line in
            (tmp_path / "qualification" / "evaluations.jsonl").read_text().splitlines()]
    fresh = next(row for row in rows if row["label"] == "endpoint-fresh")
    assert fresh["positions"] == endpoint_atoms.positions.tolist()
    assert fresh["positions"] != source.positions.tolist()
    result = json.loads((tmp_path / "qualification" / "result.json").read_text())
    assert result["status"] == "failed"
    assert "stop before Hessian" in result["error"]


def test_budget_survives_calculator_replacement(tmp_path):
    surface = BudgetSurface(QuadraticMock(), cap=5,
                            deadline=float("inf"), output=tmp_path)
    _evaluate(surface, _cell_atom(0.0), "endpoint-fresh")
    surface.replace_calculator(QuadraticMock())
    _evaluate(surface, _cell_atom(0.7), "hessian-plus")

    try:
        for x in (1.4, -0.7, 2.1, 2.8):
            _evaluate(surface, _cell_atom(x), "hessian-probe")
    except RuntimeError as error:
        assert "cap exhausted" in str(error)
    else:
        raise AssertionError("shared EFS cap was not enforced")
    assert surface.requests == 5
