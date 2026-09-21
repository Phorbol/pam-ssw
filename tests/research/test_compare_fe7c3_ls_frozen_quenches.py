import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from research.ga_ssw.compare_fe7c3_ls_frozen_quenches import (
    _combined, _frozen_bias, _norm, _pick_failed, _run_one, _softening,
)
import research.ga_ssw.compare_fe7c3_ls_frozen_quenches as diagnosis
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
from pamssw.standalone.vc_softening import FrozenPeriodicCellSoftening


class ZeroPhysical:
    def evaluate(self, atoms):
        # Deliberately nonzero physical gradient for the additive check.
        return 0.5 * float(np.sum(atoms.positions ** 2)), -atoms.positions.copy(), np.zeros((3, 3))


def test_reconstructed_ls_and_gaussian_both_change_objective_gradient():
    atoms = Atoms('Cu2', positions=[(0., 0., 0.), (1., 0., 0.)],
                  cell=np.diag([8., 8., 8.]), pbc=True)
    soft = FrozenPeriodicCellSoftening.from_atoms(
        atoms, bond_energies={(29, 29): 4.}, bond_lengths={(29, 29): 2.},
        energy_filter={(29, 29): .5})
    chart = SymmetricLogStrainChart(atoms, strain_length=8.)
    q = chart.pack(atoms); q[0] += .1
    ev = chart.evaluate(q, lambda a: _combined(ZeroPhysical(), soft, a), pressure=0.)
    physical = chart.evaluate(q, ZeroPhysical().evaluate, pressure=0.)
    assert not np.allclose(ev.gradient, physical.gradient)
    gaussian = [dict(center=np.zeros_like(q), direction=np.eye(1, len(q), 0)[0],
                     width=.6, weight=.7)]
    biased_energy, biased_gradient = _frozen_bias(
        ev.objective, chart.project(ev.gradient), q, gaussian)
    assert biased_energy > ev.objective
    assert not np.allclose(biased_gradient, ev.gradient)
    assert _norm(biased_gradient, chart.natoms) > 0.


def test_source_failed_point_reconstruction_is_exact_and_maxiter_selected():
    path = Path('research/ga_ssw/evidence/fe7c3-80-ls-filter-comparison/comparison/ls_filter-seed7/result.json')
    source = json.loads(path.read_text())
    record, climb = _pick_failed(source)
    assert record['index'] == 0
    assert climb['status'] == 'maxiter'
    assert climb['relaxation']['steps'] == 300
    assert climb['relaxation']['error'] is None
    soft = _softening(record['frozen_softening'])
    assert len(soft.pairs) == len(soft.image_shifts) == len(soft.strengths)
    q = np.asarray(climb['q'], dtype=float)
    assert q.size == 3 * len(record['chart_reference']['numbers']) + 6
    assert np.array_equal(np.asarray(record['frozen_softening']['numbers']),
                          np.asarray(record['chart_reference']['numbers']))


class ZeroCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = dict(energy=0.0, forces=np.zeros((len(atoms), 3)),
                            stress=np.zeros(6))

    def reset(self):
        self.results.clear()


@dataclass
class FakeRelaxResult:
    q: np.ndarray
    status: str = 'request_limit'


def test_run_one_uses_329_optimizer_budget_then_fresh_at_final_q(monkeypatch, tmp_path):
    source_json = json.loads(Path(
        'research/ga_ssw/evidence/fe7c3-80-ls-filter-comparison/comparison/ls_filter-seed7/result.json'
    ).read_text())
    record, climb = _pick_failed(source_json)
    from pamssw.standalone.vc_reference import VCSSWConfig
    joint = VCSSWConfig(**source_json['joint_config'])
    calls = []

    def fake_safe(q, evaluate, **kwargs):
        calls.append(kwargs['max_requests'])
        for _ in range(kwargs['max_requests']):
            evaluate(q)
        return FakeRelaxResult(np.asarray(q, dtype=float).copy())

    monkeypatch.setattr(diagnosis, 'safe_lbfgs', fake_safe)
    _run_one(dict(seed=7, arm='ls_filter', path='source.json'), record, climb,
             joint, lambda: ZeroCalculator(), tmp_path, 10, 10**20)
    result = json.loads((tmp_path / 'seed7' / 'ls_filter' / 'history10' / 'result.json').read_text())
    assert calls == [329]
    assert result['requests'] == 330
    assert result['final_fresh']['status'] == 'checked'
    assert result['safe_lbfgs']['q'] == result['q_start']
