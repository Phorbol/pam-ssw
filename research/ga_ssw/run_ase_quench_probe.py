"""Bounded ASE BFGS quench of a supplied real archive structure with original NN E/F."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ase.optimize import BFGS
from ase.io import write
from research.ga_ssw.ase_lasp_reference import LaspWaterReference,read_water_atoms


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();atoms=read_water_atoms(args.input)
    calc=LaspWaterReference(args.reference_root,args.output,args.input);atoms.calc=calc
    initial=atoms.positions.copy();start=time.monotonic()
    opt=BFGS(atoms,logfile=str(args.output/'bfgs.log'),trajectory=str(args.output/'bfgs.traj'))
    converged=bool(opt.run(fmax=1e-3,steps=30));energy=atoms.get_potential_energy()
    write(args.output/'final.extxyz',atoms)
    result=dict(input=str(args.input.resolve()),optimizer='ASE BFGS',step_cap=30,target_fmax=1e-3,
                converged=converged,steps=opt.nsteps,final_energy=energy,history=calc.history,
                max_atom_displacement=float(((atoms.positions-initial)**2).sum(axis=1).max()**.5),
                wall_seconds=time.monotonic()-start,
                scope='Force-threshold diagnostic on same uploaded NN, not independent PES or Hessian certification')
    (args.output/'quench.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))


if __name__=='__main__':main()
