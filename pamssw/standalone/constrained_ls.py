"""Small LS composition layer for fixed-substrate Cartesian SSW.

The physical calculator remains the only E/F oracle.  LS is an analytic
additive surface on the same active-coordinate chart; it is removed before a
true landing quench.
"""
from dataclasses import dataclass
import numpy as np

from .paper_reference import LSSettings
from .softening import FrozenBondSoftening
from .periodic_softening import FrozenPeriodicBondSoftening
from .generalized_numerics import safe_lbfgs


class ConstrainedLSSurface:
    def __init__(self, physical_surface, softening):
        self.physical_surface = physical_surface
        self.softening = softening

    @property
    def requests(self):
        return self.physical_surface.requests

    def evaluate(self, atoms):
        energy, forces = self.physical_surface.evaluate(atoms)
        ls_energy, ls_forces = self.softening.evaluate(atoms.copy())
        return float(energy + ls_energy), np.asarray(forces) + np.asarray(ls_forces)


@dataclass
class PreparedConstrainedLSStep:
    atoms: object
    softening: object
    energy_before: float
    energy_after: float | None
    energy_response: float | None
    optimizer: object
    requests: int


class ConstrainedLSRuntime:
    """Own one frozen LS potential and its cross-step response state."""
    def __init__(self, reference, physical_surface, softening, settings, fixed_indices):
        self.reference = reference.copy()
        self.reference.set_constraint()
        self.physical_surface = physical_surface
        self.softening = softening
        self.settings = settings
        self.fixed_indices = tuple(map(int, fixed_indices))
        self.response = None
        self.soft_surface = ConstrainedLSSurface(physical_surface, softening) if softening is not None else None

    @staticmethod
    def _domain(atoms):
        if len(atoms.pbc) != 3:
            raise ValueError('PBC must be a length-3 boolean mask')

    @classmethod
    def from_settings(cls, atoms, physical_surface, settings, *, fixed_indices):
        """Preflight settings; pair references are frozen only at true initial."""
        if not isinstance(settings, LSSettings):
            raise TypeError('settings must be LSSettings')
        cls._domain(atoms)
        reference = atoms.copy(); reference.set_constraint()
        runtime = cls(reference, physical_surface, None, settings, fixed_indices)
        from .paper_reference import LSResponseState
        runtime.response = LSResponseState(settings.target_per_atom, settings.learning_rate)
        return runtime

    def initialize_at(self, atoms):
        """Freeze pairs at the post-initial physical quench geometry."""
        clean = atoms.copy(); clean.set_constraint()
        self._domain(clean)
        if clean.pbc.any():
            softening = FrozenPeriodicBondSoftening.from_atoms(
                clean, bond_energies=self.settings.bond_energies,
                bond_lengths=self.settings.bond_lengths,
                initial_fraction=self.settings.initial_fraction, xi=self.settings.xi,
                energy_filter=self.settings.energy_filter)
        else:
            softening = FrozenBondSoftening.from_atoms(
                clean, bond_energies=self.settings.bond_energies,
                bond_lengths=self.settings.bond_lengths,
                initial_fraction=self.settings.initial_fraction, xi=self.settings.xi,
                energy_filter=self.settings.energy_filter)
        self.reference = clean
        self.softening = softening
        self.soft_surface = ConstrainedLSSurface(self.physical_surface, softening)
        return softening

    @classmethod
    def initialize(cls, atoms, physical_surface, settings, *, fixed_indices):
        if not isinstance(settings, LSSettings):
            raise TypeError('settings must be LSSettings')
        runtime = cls.from_settings(atoms, physical_surface, settings, fixed_indices=fixed_indices)
        runtime.initialize_at(atoms)
        return runtime

    def prepare(self, atoms, chart, *, fmax, max_step, maxiter, lbfgs_memory=None):
        """Soft-prequench active coordinates, while measuring true E response."""
        if getattr(getattr(self.settings, 'prequench', None), 'exit_policy', 'force') != 'force':
            raise ValueError("constrained LS preparation does not support non-default exit_policy")
        from .constrained_reference import ReducedCartesianChart
        if not isinstance(chart, ReducedCartesianChart):
            raise TypeError('chart must be ReducedCartesianChart')
        fixed = np.asarray(self.fixed_indices, dtype=int)
        if not np.array_equal(atoms.positions[fixed], self.reference.positions[fixed]):
            raise ValueError('fixed coordinates changed before LS preparation')
        before = self.physical_surface.requests
        clean = atoms.copy(); clean.set_constraint()
        true_before, true_forces = self.physical_surface.evaluate(clean)
        prequench = getattr(self.settings, "prequench", None)
        if prequench is not None:
            from .ls_cycle import LSCycleError
            active_fmax = float(np.linalg.norm(np.asarray(true_forces)[chart.active_indices], axis=1).max())
            if active_fmax > fmax:
                raise LSCycleError("true_start", f"active true force {active_fmax} exceeds fmax={fmax}")
        q0 = np.zeros(chart.dimension)
        reduced = lambda q: chart.evaluate(q, self.soft_surface)
        norm = lambda g: float(np.linalg.norm(np.asarray(g).reshape(-1, 3), axis=1).max())
        relaxed = safe_lbfgs(q0, reduced, gradient_norm=norm, step_norm=norm,
                             gtol=fmax if prequench is None else prequench.fmax, max_step=max_step,
                             maxiter=maxiter if prequench is None else prequench.steps,
                             lbfgs_memory=lbfgs_memory)
        if relaxed.energy is None or not relaxed.converged:
            return PreparedConstrainedLSStep(chart.atoms(relaxed.q), self.softening,
                float(true_before), None, None, relaxed,
                self.physical_surface.requests-before)
        prepared = chart.atoms(relaxed.q)
        try:
            true_after, _ = self.physical_surface.evaluate(prepared)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            from .ls_cycle import LSCycleError
            snapshot = PreparedConstrainedLSStep(prepared, self.softening,
                float(true_before), None, None, relaxed,
                self.physical_surface.requests-before)
            raise LSCycleError('true_response', str(error), result=snapshot) from error
        return PreparedConstrainedLSStep(prepared, self.softening, float(true_before),
            float(true_after), float((true_after-true_before)/len(atoms)), relaxed,
            self.physical_surface.requests-before)

    def update(self, next_atoms, *, energy_before, energy_after):
        clean = next_atoms.copy(); clean.set_constraint()
        new = self.response.update(self.softening, clean,
            energy_before=energy_before, energy_after=energy_after,
            bond_energies=self.settings.bond_energies,
            bond_lengths=self.settings.bond_lengths)
        self.softening = new
        self.soft_surface.softening = new
        return new


class ConstrainedNativeLSRuntime(ConstrainedLSRuntime):
    """Constrained chart adapter around the recovered Native LS runtime.

    Native initialization is deliberately deferred until ``initialize_at`` so
    the frozen bond geometry is the post-initial true-quench geometry.
    """
    def __init__(self, reference, physical_surface, settings, fixed_indices):
        self.reference = reference.copy()
        self.reference.set_constraint()
        self.physical_surface = physical_surface
        self.settings = settings
        self.fixed_indices = tuple(map(int, fixed_indices))
        self.native = None
        self.response = None
        self.softening = None
        self.soft_surface = None

    @classmethod
    def from_settings(cls, atoms, physical_surface, settings, *, fixed_indices):
        from .ls_native_reference import NativeLSSettings
        if not isinstance(settings, NativeLSSettings):
            raise TypeError('settings must be NativeLSSettings')
        cls._domain(atoms)
        return cls(atoms, physical_surface, settings, fixed_indices)

    @classmethod
    def initialize(cls, atoms, physical_surface, settings, *, fixed_indices):
        runtime = cls.from_settings(atoms, physical_surface, settings,
                                    fixed_indices=fixed_indices)
        runtime.initialize_at(atoms)
        return runtime

    def initialize_at(self, atoms):
        from .ls_native_reference import NativeLSRuntime
        clean = atoms.copy()
        clean.set_constraint()
        self._domain(clean)
        self.native = NativeLSRuntime(clean, self.settings)
        self.reference = clean
        self.softening = self.native.frozen
        self.soft_surface = ConstrainedLSSurface(self.physical_surface,
                                                  self.softening)
        return self.softening

    def update(self, next_atoms, *, energy_before, energy_after):
        if self.native is None:
            raise RuntimeError('native LS is not initialized')
        clean = next_atoms.copy()
        clean.set_constraint()
        self.native.update(self.native.frozen, clean, energy_before=energy_before,
                           energy_after=energy_after)
        self.reference = clean
        self.softening = self.native.frozen
        self.soft_surface.softening = self.softening
        return self.softening
