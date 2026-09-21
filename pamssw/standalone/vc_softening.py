"""Frozen periodic LS with exact atomic forces and affine cell stress.

Independent VC composition. Image labels and reference lengths stay fixed;
the actual cell may change. Self-image bonds contribute stress, not atomic force.
"""
import numpy as np
from .periodic_softening import FrozenPeriodicBondSoftening, _identity


class FrozenPeriodicCellSoftening(FrozenPeriodicBondSoftening):
    def __post_init__(self):
        if tuple(self.pbc) != (True, True, True):
            raise ValueError('variable-cell periodic LS requires full PBC')
        super().__post_init__()

    @classmethod
    def _build(cls, atoms, bond_energies, bond_lengths, xi, *, initial_fraction=None, total_strength=None, energy_filter=None):
        if not atoms.pbc.all():
            raise ValueError('variable-cell periodic LS requires full PBC')
        return super()._build(atoms, bond_energies, bond_lengths, xi,
                              initial_fraction=initial_fraction, total_strength=total_strength,
                              energy_filter=energy_filter)

    def _validate_atoms(self,atoms):
        numbers,_,pbc=_identity(atoms)
        if pbc != (True, True, True) or numbers!=self.numbers or pbc!=self.pbc:
            raise ValueError('ordered species/PBC must match the frozen VC-LS reference')

    def evaluate_stress(self,atoms):
        energy,forces=self.evaluate(atoms)
        delta=self._vectors(atoms);r=np.linalg.norm(delta,axis=1)
        scale=self.xi*np.asarray(self.reference_distances)
        terms=np.asarray(self.strengths)*np.exp(-(r-np.asarray(self.reference_distances))/scale)
        stress=-np.einsum('n,ni,nj->ij',terms/(scale*r),delta,delta)/atoms.get_volume()
        if not np.isfinite(stress).all():raise FloatingPointError('nonfinite LS stress')
        return energy,forces,stress


class FixedAtomicSurface:
    def __init__(self,surface):self.surface=surface
    @property
    def requests(self):return self.surface.requests
    def evaluate(self,atoms):
        e,f,_=self.surface.evaluate(atoms)
        return e,f
