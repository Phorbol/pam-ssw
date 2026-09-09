"""Observe E/F changes under permutation and rigid motion in the fixed vacuum cell.

Cell/electrostatic conventions and low-force numerical limits still require audit;
this diagnostic alone does not assign a cause to observed discrepancies.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.ga_ssw.ase_lasp_reference import LaspWaterReference,read_water_atoms


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',required=True,type=Path);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();a=read_water_atoms(args.input);c=LaspWaterReference(args.reference_root,args.output,args.input);a.calc=c
    e=a.get_potential_energy();f=a.get_forces();b=a[::-1];b.calc=c;er=b.get_potential_energy();fr=b.get_forces()[::-1]
    t=a.copy();R=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
    center=t.positions.mean(axis=0);t.positions=(t.positions-center)@R+center+np.array([1.,-2.,.5]);t.calc=c
    et=t.get_potential_energy();ft=t.get_forces()@R.T
    result=dict(reverse_energy_difference=er-e,reverse_max_force_difference=float(np.abs(fr-f).max()),
                rigid_energy_difference=et-e,rigid_max_force_difference=float(np.abs(ft-f).max()),single_point_processes=c.oracle.calls)
    (args.output/'equivariance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
