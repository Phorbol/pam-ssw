"""Edge cases for native get_atompair_v2 and Python refresh comparison."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from research.ga_ssw.probe_native_get_atompair_v2 import run
from pamssw.standalone.native_pair_selection import refresh_native_pair

def main():
 base=np.array([[0,0,0],[1.5,.4,.2],[3,0,0],[0,3,0],[0,0,3],[-2,-1,1]],float)+15
 rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
 systems=[('C-Cu-heavy',Atoms('C2Cu4',positions=base)),('C-Cu-heavy4',Atoms('C2Cu4',positions=base+np.array([.1,.2,.3]))),('C-Cu-heavy-rot90',Atoms('C2Cu4',positions=(base-15.)@rot.T+15.))]
 cases=[]
 for name,a in systems:
  matrix=[((3,2),[0.,.2,0.],'free'),((3,2),[0.,.2,.9],'free'),((4,2),[0.,.2,0.],'free'),((4,2),[0.,.2,.9],'free'),((1,0),[0.,.2,0.],'free'),((1,0),[0.,.2,0.],'one_fixed'),((1,0),[0.,.2,0.],'all_fixed'),((2,2),[0.,.2,0.],'free')]
  for pair,prefix,fixmode in matrix:
    fix={'free':np.zeros(len(a)),'one_fixed':np.array([0.,0.,1.,0.,0.,0.]),'all_fixed':np.ones(len(a))}[fixmode]
    try:
     native=run(a.positions,pair,prefix,a.numbers,fix)
     initial=tuple(i-1 if i else None for i in pair)
     py=refresh_native_pair(a,initial,iter(native['draws']),fixatom=fix)
     wanted=tuple(i-1 if i else None for i in native['pair_after'])
     row=dict(name=name,pair_input=pair,rng_prefix=prefix,fixmode=fixmode,fixatom=fix.tolist(),native=native,
              pair_match=py.pair==wanted,draw_match=py.draw_count==len(native['draws']),accepted_match=py.geometry_accepted==bool(native['events']['accepted_exit']),
              counts_match=(py.distance_or_fixatom_rejections==native['events']['distance_or_fixatom_rejections'] and py.forbidden_rejections==native['events']['forbidden_rejections'] and py.element_rejections==native['events']['element_rejections']),
              python_counts=dict(distance_or_fixatom_rejections=py.distance_or_fixatom_rejections,forbidden_rejections=py.forbidden_rejections,element_rejections=py.element_rejections))
    except Exception as e: row=dict(name=name,pair_input=pair,rng_prefix=prefix,fixatom=fix.tolist(),status='error',error=repr(e))
    cases.append(row)
 report=dict(scope='canonical get_atompair edge inputs; bounded native v2 plus Python refresh; no main/PES',case_count=len(cases),cases=cases)
 Path('research/ga_ssw/evidence/native-getpair-edges-20260917.json').write_text(json.dumps(report,indent=2)+'\n')
 completed=sum(c.get('native',{}).get('completed') is True for c in cases); matches=sum(c.get('pair_match') and c.get('draw_match') and c.get('accepted_match') and c.get('counts_match') for c in cases)
 print(json.dumps({'case_count':len(cases),'completed':completed,'errors':sum(c.get('status')=='error' for c in cases),'full_matches':matches})); assert completed==len(cases) and matches==len(cases)
if __name__=='__main__': main()
