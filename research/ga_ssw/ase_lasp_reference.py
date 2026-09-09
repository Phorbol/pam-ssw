"""Experimental ASE interface to the uploaded water NN single-point oracle.

This exposes original E/F behavior, including its numerical limitations.
Fixed vacuum cell and H30O15 composition only; no stress/cell-relax claim.
"""
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from research.ga_ssw.validate_water_archive import Oracle,read_arc


def read_water_atoms(path):
    lines,indices,symbols,xyz,_=read_arc(Path(path))
    cell=[float(v) for v in next(s for s in lines if s.startswith('PBC ')).split()[1:7]]
    return Atoms(symbols,positions=xyz,cell=cell,pbc=False)


class LaspWaterReference(Calculator):
    implemented_properties=['energy','forces']
    def __init__(self,reference_root,output,template_arc,**kwargs):
        super().__init__(**kwargs)
        self.oracle=Oracle(Path(reference_root),Path(output))
        self.lines,self.indices,_,_,_=read_arc(Path(template_arc))
        self.reference_cell=read_water_atoms(template_arc).cell.array.copy()
        self.history=[]

    def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        symbols=self.atoms.get_chemical_symbols()
        if symbols.count('H')!=30 or symbols.count('O')!=15 or len(symbols)!=45:
            raise ValueError('Reference adapter supports H30O15 only')
        if self.atoms.pbc.any() or not np.allclose(self.atoms.cell.array,self.reference_cell,rtol=0,atol=1e-10):
            raise ValueError('Reference adapter requires the fixed input vacuum cell and ASE pbc=False')
        lines=list(self.lines)
        for i,symbol,xyz in zip(self.indices,symbols,self.atoms.positions):
            fields=lines[i].split();fields[0]=fields[6]=fields[7]=symbol
            fields[1:4]=[f'{v:.12f}' for v in xyz];lines[i]=' '.join(fields)
        energy,forces,status=self.oracle.evaluate('\n'.join(lines)+'\n',f'eval-{self.oracle.calls:05d}')
        self.results={'energy':energy,'forces':forces}
        self.history.append(dict(energy=energy,max_atom_force=float(np.linalg.norm(forces,axis=1).max()),
                                 max_force_component=float(np.abs(forces).max()),wall_seconds=status['wall_seconds']))
