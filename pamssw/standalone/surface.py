"""Calculator-independent true/modified surfaces and fixed-cell quenching.

ASE units: energy eV, positions Angstrom, forces eV/Angstrom. Requests
count API evaluations, not SCF iterations or backend internal force calls.
One ASESurface owns one calculator and is intended for serial use; parallel
walkers must receive separate calculators (and directories for file backends).
"""
from dataclasses import dataclass

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS


class ASESurface:
    """Select the calculator objective explicitly when electronic smearing matters.

Default energy follows ASE's ordinary get_potential_energy convention.
For calculators whose forces differentiate free_energy instead, pass
force_consistent=True; missing support is propagated, never silently
replaced with a different objective. This interface cannot itself prove
that a backend's supplied energy and forces are mathematically consistent.
"""
    def __init__(self, calculator, *, force_consistent=False):
        if calculator is None:
            raise ValueError('an ASE calculator is required')
        self.calculator = calculator
        self.force_consistent = bool(force_consistent)
        self.requests = 0

    def evaluate(self, atoms):
        if atoms.constraints:
            raise NotImplementedError('constraint geometry is not implemented by this surface')
        if not len(atoms) or not np.isfinite(atoms.positions).all():
            raise ValueError('finite nonempty geometry required')
        work = atoms.copy()
        work.calc = self.calculator
        self.requests += 1  # Failed requests count too.
        energy = float(work.get_potential_energy(force_consistent=self.force_consistent))
        forces = np.array(work.get_forces(), dtype=float, copy=True)
        if not np.isfinite(energy) or forces.shape != (len(work), 3) or not np.isfinite(forces).all():
            raise ValueError('calculator returned invalid energy or forces')
        return energy, forces


class SurfaceCalculator(Calculator):
    """Evaluate a fixed collection of additive terms on a true ASE surface.

Terms provide evaluate(atoms)->(energy, forces). Construct a new calculator
when terms change; mutating a term behind ASE's cache is unsupported.
free_energy names the SAME selected objective as energy, so an optimizer
cannot silently switch from the user's explicit true-surface choice.
"""
    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, surface, *, terms=()):
        super().__init__()
        self.surface = surface
        self.terms = tuple(terms)

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces = self.surface.evaluate(self.atoms)
        true_energy, true_forces = energy, forces.copy()
        for term in self.terms:
            de, df = term.evaluate(self.atoms)
            df = np.asarray(df, dtype=float)
            if not np.isfinite(de) or df.shape != forces.shape or not np.isfinite(df).all():
                raise ValueError('bias returned invalid energy or forces')
            energy += float(de)
            forces += df
        self.results = dict(energy=energy, free_energy=energy, forces=forces,
                            true_energy=true_energy, true_forces=true_forces)


@dataclass(frozen=True)
class QuenchResult:
    atoms: object
    energy: float
    max_force: float
    converged: bool
    optimizer_steps: int
    evaluation_requests: int
    surface: str


def quench(atoms, surface, *, fmax, steps, terms=(), optimizer=BFGS):
    """Relax at fixed cell, returning an explicit force certificate.

With terms=() this certifies only force stationarity on the true surface,
not a positive Hessian or physical stability. With terms supplied this is
a modified-surface quench; it must not be archived as a true minimum.
Caller geometry/calculator are untouched, and output has no calculator.
"""
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError('fmax must be positive and finite')
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    terms = tuple(terms)
    work = atoms.copy()
    work.calc = SurfaceCalculator(surface, terms=terms)
    before = surface.requests
    opt = optimizer(work, logfile=None)
    opt.run(fmax=fmax, steps=int(steps))
    forces = work.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max())
    energy = float(work.get_potential_energy())
    output = work.copy()
    return QuenchResult(output, energy, max_force, max_force <= fmax,
                        opt.nsteps, surface.requests - before,
                        'modified' if terms else 'true')
