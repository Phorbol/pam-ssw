from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pamssw.calculators import EnergyResult
from pamssw.config import LSSSWConfig, SSWConfig
from pamssw.runner import run_ls_ssw, run_ssw
from pamssw.state import State


@dataclass
class VolumeWellCalculator:
    target_volume: float = 64.0
    k_volume: float = 0.01
    k_fractional: float = 0.1

    def evaluate(self, state: State) -> EnergyResult:
        frac = state.positions @ np.linalg.inv(state.cell)
        volume = float(np.linalg.det(state.cell))
        energy = 0.5 * self.k_volume * (volume - self.target_volume) ** 2
        energy += 0.5 * self.k_fractional * float(np.sum((frac - 0.25) ** 2))
        grad_frac = self.k_fractional * (frac - 0.25)
        grad_cart = grad_frac @ np.linalg.inv(state.cell).T
        stress = self.k_volume * (volume - self.target_volume) * np.eye(3)
        return EnergyResult(energy=energy, gradient=grad_cart, stress=stress)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        result = self.evaluate(template.with_flat_positions(flat_positions))
        return result.energy, result.gradient.reshape(-1)


def compressed_periodic_state() -> State:
    cell = np.eye(3) * 3.6
    frac = np.array([[0.2, 0.2, 0.2], [0.55, 0.2, 0.2], [0.2, 0.55, 0.2]])
    return State(
        numbers=np.array([6, 6, 6]),
        positions=frac @ cell,
        cell=cell,
        pbc=(True, True, True),
    )


def test_vc_ssw_smoke_updates_cell_stats() -> None:
    config = SSWConfig(
        max_trials=1,
        max_steps_per_walk=1,
        oracle_candidates=1,
        proposal_relax_steps=5,
        quench_maxiter=5,
        coordinate_mode="variable_cell",
        cell_dof_mode="volume_only",
        finite_diff_cell_gradient=False,
        variable_cell_requires_stress=True,
        n_bond_pairs=0,
        n_cell_random_candidates=1,
        n_coupled_random_candidates=1,
    )
    result = run_ssw(compressed_periodic_state(), VolumeWellCalculator(), config)
    assert result.stats["variable_cell_supported"] == 1
    assert result.stats["coordinate_system"] == "generalized_fractional_log_deformation"
    assert result.stats["cell_dof_mode"] == "volume_only"
    assert result.best_state.cell is not None


def test_vc_ls_ssw_smoke_keeps_local_softening_available() -> None:
    config = LSSSWConfig(
        max_trials=1,
        max_steps_per_walk=1,
        oracle_candidates=1,
        proposal_relax_steps=3,
        quench_maxiter=3,
        coordinate_mode="variable_cell",
        cell_dof_mode="volume_only",
        finite_diff_cell_gradient=False,
        variable_cell_requires_stress=True,
        n_bond_pairs=0,
        n_cell_random_candidates=1,
        n_coupled_random_candidates=1,
        local_softening_mode="neighbor_auto",
        local_softening_strength=0.1,
    )
    result = run_ls_ssw(compressed_periodic_state(), VolumeWellCalculator(), config)
    assert result.stats["variable_cell_supported"] == 1
    assert result.stats["local_softening_builds"] >= 0
