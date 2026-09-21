"""Expanded frozen input set for the bounded native get_atompair probe."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import molecule
from research.ga_ssw.probe_native_get_atompair_v2 import run

def custom(symbols, xyz): return Atoms(symbols, positions=np.asarray(xyz,float))

def main():
    cases=[]
    systems=[('C2H6',molecule('C2H6')),('CH3OH',molecule('CH3OH')),('C6H6',molecule('C6H6'))]
    systems += [
      ('Cu13',custom('Cu13',[[0,0,0],[2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2],[1.4,1.4,1.4],[-1.4,1.4,-1.4],[1.4,-1.4,-1.4],[-1.4,-1.4,1.4],[3.5,.5,-.3],[-3.5,-.5,.3]])),
      ('AuAg-mixed',custom('Au3Ag3',[[0,0,0],[2.5,0,0],[0,2.5,0],[0,0,2.5],[2,2,0],[-2,0,1.]])),
      ('C-Cu-mixed',custom('C2Cu4',[[0,0,0],[1.5,.4,.2],[3,0,0],[0,3,0],[0,0,3],[-2,-1,1.]])),
      ('H2',custom('H2',[[0,0,0],[1.0,.1,0.]])),
      ('H3',custom('H3',[[0,0,0],[1.1,0,0],[.4,.9,.2]])),
    ]
    for name,atoms in systems:
      for label,prefix in [('preserve-first',[0.,.2,0.]),('replace-first',[.9,.7,0.]),('neighbor-first',[0.,.2,.9])]:
        try:
          canonical_positions=np.asarray(atoms.positions,float)+15.0
          assert np.all(canonical_positions > 0.0) and np.all(canonical_positions < 30.0)
          assert np.max(np.ptp(canonical_positions,axis=0)) < 15.0
          row=run(canonical_positions,(1,2),prefix,atoms.numbers,np.zeros(len(atoms)))
          row['canonical_positions']=canonical_positions.tolist()
          row.update(name=name,branch=label,status='completed' if row.get('completed') else 'incomplete')
        except Exception as exc:
          row=dict(name=name,branch=label,status='error',error=repr(exc))
        cases.append(row)
    report=dict(scope='canonical expanded fixed inputs; all coordinates shifted +15; bounded get_atompair_v2; no main/PES',case_count=len(cases),cases=cases)
    Path('research/ga_ssw/evidence/native-getpair-canonical-20260917.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'case_count':len(cases),'completed':sum(c.get('completed',False) is True for c in cases),'errors':sum(c.get('status')=='error' for c in cases)}))
    assert all(c.get('status')=='completed' and c.get('completed') is True for c in cases)
if __name__=='__main__': main()
