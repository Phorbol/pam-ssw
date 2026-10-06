import json
import socket
import tempfile
import threading
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from research.ga_ssw.lasp_external_ase import _Service


class FailingCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.calls += 1
        raise RuntimeError("synthetic calculator failure")


class CountingCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {"energy": -1.0, "forces": np.zeros((len(atoms), 3))}


def _request(server_path, coord):
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(server_path))
        client.sendall((json.dumps({"coord": coord}) + "\n").encode())
        return json.loads(client.makefile("r").readline())


def _coord():
    return "10 0 0\n0 10 0\n0 0 10\nCu 1 1 1 1\n"


def _serve_requests(calculator, coords, max_requests=1):
    template = Atoms("Cu", positions=[[1, 1, 1]], cell=np.eye(3) * 10, pbc=True)
    with tempfile.TemporaryDirectory(prefix="lasp-budget-", dir=Path.home()) as tmp:
        path = Path(tmp) / "s"
        server = _Service(str(path), calculator, template, [True] * 3, max_requests)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            replies = [_request(path, coord) for coord in coords]
            return replies, list(server.log), list(server.errors)
        finally:
            server.shutdown()
            worker.join()
            server.server_close()


def test_failed_calculator_attempt_consumes_request_cap():
    calculator = FailingCalculator()

    replies, successful, errors = _serve_requests(calculator, [_coord(), _coord()], max_requests=1)

    assert calculator.calls == 1
    assert successful == []
    assert len(errors) == 2
    assert replies[0]["ok"] is False
    assert "synthetic calculator failure" in replies[0]["error"]
    assert replies[1]["ok"] is False
    assert "external request cap reached" in replies[1]["error"]


def test_successful_requests_preserve_calculator_cache_behavior():
    calculator = CountingCalculator()

    replies, successful, errors = _serve_requests(calculator, [_coord(), _coord()], max_requests=2)

    assert calculator.calls == 1
    assert len(successful) == 2
    assert errors == []
    assert [reply["ok"] for reply in replies] == [True, True]


def test_geometry_rejection_does_not_consume_calculation_cap():
    calculator = CountingCalculator()
    wrong_cell = "11 0 0\n0 10 0\n0 0 10\nCu 1 1 1 1\n"

    replies, successful, errors = _serve_requests(
        calculator, [wrong_cell, _coord(), _coord()], max_requests=1)

    assert calculator.calls == 1
    assert len(successful) == 1
    assert len(errors) == 2
    assert "fixed cell" in replies[0]["error"]
    assert replies[1]["ok"] is True
    assert "external request cap reached" in replies[2]["error"]
