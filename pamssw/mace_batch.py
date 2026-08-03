from __future__ import annotations

from dataclasses import dataclass, field
import threading

import numpy as np

from .calculators import ASECalculator, EnergyResult
from .state import State


@dataclass
class MACEBatchCalculator:
    """Explicit MACE graph batching for independent force evaluations.

    The wrapper keeps the ordinary ASE path for single structures.  Only callers
    that deliberately submit ``evaluate_flat_many`` use MACE graph batching.
    Each submitted structure remains one physical force evaluation for budget
    accounting.
    """

    calculator: object
    _serial: ASECalculator = field(init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._serial = ASECalculator(self.calculator)

    @property
    def supports_batch_evaluation(self) -> bool:
        return True

    def evaluate(self, state: State) -> EnergyResult:
        with self._lock:
            return self._serial.evaluate(state)

    def evaluate_flat(
        self,
        flat_positions: np.ndarray,
        template: State,
    ) -> tuple[float, np.ndarray]:
        with self._lock:
            return self._serial.evaluate_flat(flat_positions, template)

    def evaluate_flat_many(
        self,
        flat_positions: tuple[np.ndarray, ...],
        templates: tuple[State, ...],
    ) -> tuple[tuple[float, np.ndarray], ...]:
        positions = tuple(np.asarray(value, dtype=float) for value in flat_positions)
        state_templates = tuple(templates)
        if not positions:
            raise ValueError("batch evaluation requires at least one geometry")
        if len(positions) != len(state_templates):
            raise ValueError("flat_positions and templates must have the same length")
        states = tuple(
            template.with_flat_positions(value)
            for value, template in zip(positions, state_templates)
        )
        reference_pbc = states[0].pbc
        if any(state.pbc != reference_pbc for state in states[1:]):
            raise ValueError("one MACE graph batch requires identical periodicity")
        atoms = tuple(self._serial._to_atoms(state) for state in states)
        with self._lock:
            energies, forces = self._evaluate_atoms_many(atoms)
        return tuple(
            (
                float(energy),
                -np.asarray(force, dtype=float).reshape(-1),
            )
            for energy, force in zip(energies, forces)
        )

    def _evaluate_atoms_many(self, atoms_batch) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        import torch
        from mace import data
        from mace.tools import torch_geometric

        self._set_les_periodicity(atoms_batch[0])
        self.calculator.arrays_keys.update(
            {self.calculator.charges_key: "charges"}
        )
        keyspec = data.KeySpecification(
            info_keys=self.calculator.info_keys,
            arrays_keys=self.calculator.arrays_keys,
        )
        atomic_data = []
        for atoms in atoms_batch:
            config = data.config_from_atoms(
                atoms,
                key_specification=keyspec,
                head_name=self.calculator.head,
            )
            atomic_data.append(
                data.AtomicData.from_config(
                    config,
                    z_table=self.calculator.z_table,
                    cutoff=self.calculator.r_max,
                    heads=self.calculator.available_heads,
                )
            )
        batch = torch_geometric.Batch.from_data_list(atomic_data).to(
            self.calculator.device
        )
        model_energies = []
        model_forces = []
        for model in self.calculator.models:
            model_batch = self.calculator._clone_batch(batch)
            output = model(
                model_batch.to_dict(),
                compute_stress=not self.calculator.use_compile,
                training=self.calculator.use_compile,
                compute_edge_forces=self.calculator.compute_atomic_stresses,
                compute_atomic_stresses=self.calculator.compute_atomic_stresses,
            )
            model_energies.append(output["energy"].detach())
            model_forces.append(output["forces"].detach())
        energies = (
            torch.stack(model_energies).mean(dim=0).cpu().numpy()
            * self.calculator.energy_units_to_eV
        )
        flat_forces = (
            torch.stack(model_forces).mean(dim=0).cpu().numpy()
            * self.calculator.energy_units_to_eV
        )
        forces = tuple(
            flat_forces[int(batch.ptr[index]) : int(batch.ptr[index + 1])]
            for index in range(len(atoms_batch))
        )
        return np.asarray(energies, dtype=float), forces

    def _set_les_periodicity(self, atoms) -> None:
        pbc = tuple(bool(value) for value in np.asarray(atoms.get_pbc()))
        periodic_dimensions = sum(pbc)
        for model in self.calculator.models:
            if not hasattr(model, "les") or not hasattr(model.les, "ewald"):
                continue
            if periodic_dimensions == 2:
                model.les.ewald.periodicity = "2d"
                model.les.ewald.slab_axis = next(
                    axis for axis, periodic in enumerate(pbc) if not periodic
                )
            elif periodic_dimensions == 3:
                model.les.ewald.periodicity = "3d"
