"""Reoptimize public rho6 reference shapes at the paper's rho14; no search."""
from pathlib import Path
import json
import numpy as np
from ase import Atoms
from ase.calculators.morse import MorsePotential
from ase.optimize import LBFGSLineSearch
from ase.io import write
HERE=Path(__file__).resolve().parent
rows=[]
for label,n,target in [('29E',29,-102.774589),('80G',80,-340.811371)]:
    coords=np.loadtxt(HERE/'references'/f'morse-{label}.points',skiprows=1)
    assert coords.shape==(n,3)
    atoms=Atoms(f'Ar{n}',positions=2.7*coords)
    # At these compact geometries all pairs stay below the inner cutoff.
    # This is the full Morse formula here, not the default ASE truncated model.
    calc=MorsePotential(epsilon=1.,rho0=14.,r0=2.7,rcut1=100.,rcut2=101.)
    atoms.calc=calc
    initial=float(atoms.get_potential_energy())
    max_distance=[float(atoms.get_all_distances().max())]
    def check_domain():
        diameter=float(atoms.get_all_distances().max())
        max_distance.append(diameter)
        if diameter>=270.: raise RuntimeError('Morse cutoff entered')
    opt=LBFGSLineSearch(atoms,logfile=str(HERE/f'morse-{label}.log'))
    opt.attach(check_domain)
    converged=opt.run(fmax=1e-4,steps=500)
    energy=float(atoms.get_potential_energy())
    fmax=float(np.linalg.norm(atoms.get_forces(),axis=1).max())
    write(HERE/'references'/f'morse-{label}-rho14.extxyz',atoms)
    rows.append(dict(label=label,n=n,initial_rho14_energy=initial,energy=energy,target=target,
                     delta=energy-target,fmax=fmax,steps=opt.nsteps,converged=bool(converged),
                     max_diameter=max(max_distance),qualified=bool(converged and abs(energy-target)<1e-5)))
(HERE/'morse-reference-qualification.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
