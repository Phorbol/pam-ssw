"""Small, explicit adapter for ASE FixAtoms and Hookean constraints.

The physical surface remains a constraint-free oracle.  Hookean corrections are
evaluated once through ASE's own constraint methods; fixed-atom projection is
left to the caller/ASE and is attached after Hookean constraints.
"""
from dataclasses import dataclass
import copy
import json
import numbers

import numpy as np
from ase.constraints import FixAtoms, Hookean


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    return value


def _hookean_from_spec(spec):
    data = json.loads(spec) if isinstance(spec, str) else copy.deepcopy(spec)
    if data.get('name') != 'Hookean' or set(data) != {'name', 'kwargs'}:
        raise ValueError('invalid canonical Hookean specification')
    kwargs = data['kwargs']
    a2 = kwargs.get('a2')
    if isinstance(a2, list):
        a2 = tuple(a2)
    return Hookean(kwargs['a1'], a2, kwargs['k'], kwargs.get('rt'))


def _canonical_hookean(constraint, n):
    if not isinstance(constraint, Hookean):
        raise TypeError('only FixAtoms and Hookean constraints are supported')
    data = _jsonable(constraint.todict())
    kwargs = data['kwargs']
    a1 = kwargs.get('a1')
    if (isinstance(a1, bool) or not isinstance(a1, numbers.Integral) or
            not 0 <= int(a1) < n):
        raise ValueError('Hookean atom index is out of range')
    a2 = kwargs.get('a2')
    if isinstance(a2, (int, np.integer)) and not isinstance(a2, bool):
        if not 0 <= int(a2) < n:
            raise ValueError('Hookean atom index is out of range')
    elif isinstance(a2, list) and len(a2) in (3, 4):
        if not np.isfinite(np.asarray(a2, dtype=float)).all():
            raise ValueError('Hookean point/plane must be finite')
        if len(a2) == 4 and np.linalg.norm(np.asarray(a2[:3], dtype=float)) == 0:
            raise ValueError('Hookean plane normal must be nonzero')
    else:
        raise ValueError('invalid Hookean target')
    k = kwargs.get('k')
    if isinstance(k, bool) or not np.isscalar(k) or not np.isfinite(k) or k < 0:
        raise ValueError('Hookean spring constant must be finite and nonnegative')
    rt = kwargs.get('rt')
    if rt is None and not (isinstance(a2, list) and len(a2) == 4):
        raise ValueError('pair/point Hookean constraints require a threshold rt')
    if rt is not None and (isinstance(rt, bool) or not np.isscalar(rt) or
                           not np.isfinite(rt) or rt < 0):
        raise ValueError('Hookean threshold must be finite and nonnegative')
    if isinstance(a2, list) and len(a2) == 4 and rt is not None:
        raise ValueError('plane Hookean constraints must omit rt')
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


@dataclass(frozen=True)
class ConstraintSet:
    fixed_indices: tuple = ()
    hookean_specs: tuple = ()

    def clean_atoms(self, atoms):
        clean = atoms.copy()
        clean.set_constraint()
        return clean

    def attach(self, atoms):
        attached = atoms.copy()
        constraints = [_hookean_from_spec(spec) for spec in self.hookean_specs]
        if self.fixed_indices:
            constraints.append(FixAtoms(indices=list(self.fixed_indices)))
        attached.set_constraint(constraints)
        return attached


def normalize_constraints(atoms, fixed_indices=None):
    n = len(atoms)
    fixed = set()
    present = set()
    if fixed_indices is not None:
        for index in fixed_indices:
            if isinstance(index, bool) or not isinstance(index, numbers.Integral) or not 0 <= int(index) < n:
                raise ValueError('fixed_indices must contain valid atom indices')
            fixed.add(int(index))
    specs = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            present.update(int(i) for i in constraint.get_indices())
        elif isinstance(constraint, Hookean):
            specs.append(_canonical_hookean(constraint, n))
        else:
            raise TypeError('only FixAtoms and Hookean constraints are supported')
    if fixed_indices is not None and present and fixed != present:
        raise ValueError('explicit fixed indices must agree with FixAtoms')
    fixed.update(present)
    if any(i < 0 or i >= n for i in fixed):
        raise ValueError('FixAtoms indices must be valid for this structure')
    return ConstraintSet(tuple(sorted(fixed)), tuple(specs))


class HookeanSurface:
    """Constraint correction wrapper around a constraint-free physical surface."""
    def __init__(self, physical_surface, specs):
        if isinstance(specs, ConstraintSet):
            if specs.fixed_indices:
                raise ValueError('HookeanSurface accepts Hookean specs; FixAtoms projection is caller-owned')
            specs = specs.hookean_specs
        self.physical_surface = physical_surface
        self.specs = tuple(specs)
        self._constraints = tuple(_hookean_from_spec(spec) for spec in self.specs)
        self.last_evaluation = None

    @property
    def requests(self):
        return self.physical_surface.requests

    @property
    def exhausted(self):
        return getattr(self.physical_surface, 'exhausted', False)

    def evaluate(self, atoms):
        self.last_evaluation = None
        clean = atoms.copy()
        clean.set_constraint()
        physical_energy, physical_forces = self.physical_surface.evaluate(clean)
        physical_forces = np.asarray(physical_forces, dtype=float)
        if physical_forces.shape != (len(clean), 3):
            raise ValueError('physical surface returned invalid forces')
        constrained = clean.copy()
        constrained.set_constraint([copy.deepcopy(c) for c in self._constraints])
        hookean_forces = np.zeros_like(physical_forces)
        hookean_energy = 0.0
        for constraint in constrained.constraints:
            hookean_energy += float(constraint.adjust_potential_energy(constrained))
            constraint.adjust_forces(constrained, hookean_forces)
        total_energy = float(physical_energy) + hookean_energy
        total_forces = physical_forces + hookean_forces
        if not np.isfinite(total_energy) or not np.isfinite(total_forces).all():
            raise ValueError('Hookean surface returned nonfinite energy or forces')
        self.last_evaluation = dict(physical_energy=float(physical_energy),
            physical_forces=physical_forces.copy(), hookean_energy=hookean_energy,
            hookean_forces=hookean_forces.copy(), energy=total_energy,
            forces=total_forces.copy())
        return total_energy, total_forces


def bind_hookean_surface(surface, specs):
    specs = tuple(specs.hookean_specs if isinstance(specs, ConstraintSet) else specs)
    if not specs:
        return surface
    if isinstance(surface, HookeanSurface):
        if surface.specs == specs:
            return surface
        raise ValueError('cannot stack different HookeanSurface wrappers')
    return HookeanSurface(surface, specs)
