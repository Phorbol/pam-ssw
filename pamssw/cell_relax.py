"""True E+pV quenching with physical force and allowed-stress certificates."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

from .result import RelaxOutcomeClass, RelaxResult, RelaxTelemetry
from .state import State


def _validate_cell(cell):
    if cell is None or not np.all(np.isfinite(cell)):
        raise ValueError('cell must be finite and nonsingular with positive volume')
    volume = float(np.linalg.det(cell))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError('cell must be nonsingular with positive volume')


def _state_from_atoms(atoms, template, metadata=None):
    return State(template.numbers.copy(), atoms.positions.copy(),
                 np.asarray(atoms.cell).copy(), tuple(atoms.pbc),
                 template.fixed_mask.copy(),
                 template.metadata.copy() if metadata is None else metadata)


class _CellCalculator(Calculator):
    """ASE cache around exactly one PAM evaluation per changed geometry."""
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, calculator, template):
        super().__init__()
        self.calculator = calculator
        self.template = template
        self.evaluator_calls = 0

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        state = _state_from_atoms(atoms, self.template)
        _validate_cell(state.cell)
        if not np.all(np.isfinite(state.positions)):
            raise ValueError('atomic positions must be finite')
        result = self.calculator.evaluate(state)
        self.evaluator_calls += 1
        if result.stress is None:
            raise ValueError('cell relaxation requires calculator stress')
        stress = np.asarray(result.stress, dtype=float)
        if stress.shape == (6,):
            stress = voigt_6_to_full_3x3_stress(stress)
        if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
            raise ValueError('calculator stress must be a finite 3x3 tensor or Voigt 6-vector')
        if not np.allclose(stress, stress.T, atol=1e-10, rtol=1e-7):
            raise ValueError('calculator stress must be symmetric')
        gradient = np.asarray(result.gradient, dtype=float)
        if gradient.size != state.n_atoms * 3 or not np.all(np.isfinite(gradient)):
            raise ValueError('calculator atomic gradient must be finite and match the atoms')
        if not np.isfinite(result.energy):
            raise ValueError('calculator energy must be finite')
        self.results = dict(energy=float(result.energy), free_energy=float(result.energy),
                            forces=-gradient.reshape(-1, 3),
                            stress=full_3x3_to_voigt_6_stress(stress))


class _PhysicalConvergence:
    """Cover both current and earlier ASE optimizer convergence entry points."""
    def converged(self, *args, **kwargs):
        return self.physical_convergence()

    def gradient_converged(self, *args, **kwargs):
        return self.physical_convergence()


class _PhysicalLBFGS(_PhysicalConvergence, LBFGS):
    pass


class _PhysicalFIRE(_PhysicalConvergence, FIRE):
    pass


class CellRelaxer:
    """Relax atoms and selected cell degrees of freedom on E+pV.

    Stress uses ASE's tensile-positive convention, in eV/Angstrom^3. The
    certificate is max absolute allowed Cauchy-stress component (mean normal
    stress for volume_only), after adding external pressure. Fixed atoms keep
    fractional coordinates under the affine deformation. slab_xy requires an
    XY plane and fixes the third cell vector; its stress uses the full cell
    volume, so its numerical tolerance depends on vacuum thickness.
    """
    def __init__(self, calculator, optimizer='ase-lbfgs', mode='shape',
                 pressure_gpa=0.0, stress_tol=1e-3):
        if optimizer not in {'ase-lbfgs', 'ase-fire'}:
            raise ValueError('cell relaxation requires ase-lbfgs or ase-fire')
        if mode not in {'shape', 'volume_only', 'slab_xy'}:
            raise ValueError('unknown cell relaxation mode')
        if not np.isfinite(pressure_gpa):
            raise ValueError('external pressure must be finite')
        if not np.isfinite(stress_tol) or stress_tol <= 0:
            raise ValueError('stress_tol must be positive and finite')
        if mode == 'slab_xy' and pressure_gpa != 0:
            raise ValueError('nonzero external pressure requires bulk periodic geometry')
        self.calculator = calculator
        self.optimizer = optimizer
        self.mode = mode
        self.pressure_gpa = float(pressure_gpa)
        self.pressure = self.pressure_gpa * units.GPa
        self.stress_tol = float(stress_tol)

    def _make_atoms(self, state):
        _validate_cell(state.cell)
        if state.n_atoms == 0:
            raise ValueError('cell relaxation requires at least one atom')
        if self.mode == 'slab_xy':
            if state.pbc not in {(True, True, False), (True, True, True)}:
                raise ValueError('slab_xy requires periodic X and Y')
            if not (np.allclose(state.cell[:2, 2], 0, atol=1e-12, rtol=0)
                    and np.allclose(state.cell[2, :2], 0, atol=1e-12, rtol=0)):
                raise ValueError('slab_xy requires an axis-aligned XY cell and Z vacuum vector')
        elif not all(state.pbc):
            raise ValueError('bulk cell relaxation requires full periodic PBC')
        atoms = Atoms(numbers=state.numbers, positions=state.positions,
                      cell=state.cell, pbc=state.pbc)
        if np.any(state.fixed_mask):
            atoms.set_constraint(FixAtoms(mask=state.fixed_mask))
        atoms.calc = _CellCalculator(self.calculator, state)
        return atoms

    def _physical_norms(self, atoms):
        forces = atoms.get_forces()
        force_norm = float(np.max(np.linalg.norm(forces, axis=1), initial=0))
        residual = atoms.get_stress(voigt=False) + self.pressure * np.eye(3)
        if self.mode == 'volume_only':
            stress_norm = float(abs(np.trace(residual) / 3))
        elif self.mode == 'slab_xy':
            stress_norm = float(np.max(np.abs(residual[:2, :2])))
        else:
            stress_norm = float(np.max(np.abs(residual)))
        return force_norm, stress_norm

    def relax(self, state: State, fmax: float, maxiter: int,
              trajectory_callback: Callable[[State], None] | None = None,
              trajectory_stride: int = 1) -> RelaxResult:
        if not np.isfinite(fmax) or fmax <= 0:
            raise ValueError('fmax must be positive and finite')
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 0:
            raise ValueError('maxiter must be a nonnegative integer')
        if (isinstance(trajectory_stride, bool) or not isinstance(trajectory_stride, int)
                or trajectory_stride < 1):
            raise ValueError('trajectory_stride must be a positive integer')
        atoms = self._make_atoms(state)
        filt = FrechetCellFilter(atoms,
                                 mask=[1, 1, 0, 0, 0, 1] if self.mode == 'slab_xy' else None,
                                 hydrostatic_strain=self.mode == 'volume_only',
                                 scalar_pressure=self.pressure)
        optimizer_cls = _PhysicalLBFGS if self.optimizer == 'ase-lbfgs' else _PhysicalFIRE
        optimizer = optimizer_cls(filt, logfile=None)

        def physical_convergence():
            force, stress = self._physical_norms(atoms)
            return bool(force <= fmax and stress <= self.stress_tol)

        optimizer.physical_convergence = physical_convergence
        if trajectory_callback is not None:
            optimizer.attach(lambda: trajectory_callback(_state_from_atoms(atoms, state)),
                             interval=trajectory_stride)
        converged = bool(optimizer.run(fmax=fmax, steps=maxiter))
        force_norm, stress_norm = self._physical_norms(atoms)
        energy = float(atoms.get_potential_energy())
        volume = float(atoms.get_volume())
        enthalpy = energy + self.pressure * volume
        metadata = dict(state.metadata, potential_energy=energy, enthalpy=enthalpy,
                        volume=volume, external_pressure_gpa=self.pressure_gpa,
                        stress_norm=stress_norm, quench_cell_mode=self.mode,
                        stress=atoms.get_stress(voigt=False).copy())
        relaxed = _state_from_atoms(atoms, state, metadata)
        if trajectory_callback is not None:
            trajectory_callback(relaxed)
        # Fractional displacement removes the imposed affine cell change.
        reference_affine = state.positions @ np.linalg.solve(state.cell, relaxed.cell)
        norms = np.linalg.norm((relaxed.positions-reference_affine)[state.movable_mask], axis=1)
        displacement_rms = float(np.sqrt(np.mean(norms**2))) if norms.size else 0.0
        displacement_max = float(np.max(norms, initial=0))
        return RelaxResult(
            state=relaxed, energy=enthalpy, gradient_norm=force_norm,
            n_iter=optimizer.nsteps, potential_energy=energy, volume=volume,
            stress_norm=stress_norm, displacement_rms=displacement_rms,
            displacement_max=displacement_max,
            outcome_class=(RelaxOutcomeClass.CONVERGED_PRODUCTIVE if converged
                           else RelaxOutcomeClass.USEFUL_PROGRESS),
            telemetry=RelaxTelemetry(
                backend=self.optimizer, evaluator_calls=atoms.calc.evaluator_calls,
                backend_evaluations=atoms.calc.evaluator_calls,
                gradient_measure='raw_active_max_force_and_allowed_stress',
                converged=converged, optimizer_success=converged,
                termination_reason='converged' if converged else 'maxiter',
                accepted_steps=optimizer.nsteps))
