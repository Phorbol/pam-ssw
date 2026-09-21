"""Synthetic public-entry check for the research adapter."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pamssw.standalone.paper_reference import SSWConfig, run_ssw
from pamssw.standalone.surface import ASESurface
from research.ga_ssw.native_stage_quench_adapter import StatefulNativeStageAdapter


class _Harmonic(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float((atoms.positions ** 2).sum() / 2),
            "forces": -atoms.positions.copy(),
        }


def test_public_run_ssw_release_all_reaches_true_landing():
    config = SSWConfig(
        width=0.2, rotation_bias=2.0, max_gaussians=1, temperature_K=300.0,
        fmax=1e-4, relax_steps=10, fd_step=1e-3, rotation_hvp=8,
        rotation_tol=1e-5, direction_sampling="global",
    )
    adapter = StatefulNativeStageAdapter(
        predicate_kwargs={
            "climb_stopf": -1.0, "maxe_height": 0.0,
            "e_maxlimit": 99.0, "f_maxlimit": 99.0,
            "e_maxlimit_gm": 99.0, "ngaus_relax": 10,
            "ngaus_relax_ini": 10, "multi_pes": False,
            "counter_start": 1,
        },
        stop_on="allstop", energy_reference="current", gm_reference="best",
    )
    surface = ASESurface(_Harmonic())
    result = run_ssw(
        Atoms("H", positions=[[0.1, 0.2, 0.3]]), surface,
        steps=1, config=config, rng=np.random.default_rng(9),
        bias_quench_adapter=adapter,
    )
    record = result.records[0]
    assert record.climb[0]["release_all"] is True
    assert record.landing is not None
    assert result.minima[-1].surface == "true"
    assert result.minima[-1].converged
    assert result.initial.evaluation_requests + sum(
        item.evaluation_requests for item in result.records
    ) == surface.requests
