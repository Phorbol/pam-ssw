"""Runner wiring check with an analytic surface; no material-performance claim."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('type4_runner', HERE / 'qualify_type4.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_type4_complete_runner_with_counted_harmonic_oracle(tmp_path, monkeypatch):
    import mace.calculators
    from ase.io import read
    reference = read(runner.SOURCE / 'input.arc', index=0).positions.copy()

    class Harmonic(Calculator):
        implemented_properties = ['energy', 'forces']
        def __init__(self, **kwargs):
            super().__init__()
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            delta = atoms.positions - reference
            # Nondegenerate Hessian exercises rotation; isotropic Hv is collinear.
            stiffness = np.linspace(1., 3., delta.size).reshape(delta.shape)
            self.results = {'energy': .5 * np.sum(stiffness * delta**2),
                            'forces': -stiffness * delta}

    monkeypatch.setattr(mace.calculators, 'MACECalculator', Harmonic)
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['qualify_type4.py', '--output', str(out)])
    runner.main()
    import json
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['qualified'], summary
    assert summary['live_total'] > 0
    assert summary['replay_total'] > 0
    assert summary['fresh_total'] == 6
