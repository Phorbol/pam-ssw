"""Locate continuous/first-leg divergence from saved records; zero PES."""
import json,pickle
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent/'mace-v1'
def read(p):
 with p.open('rb') as f:return pickle.load(f)
def delta(a,b):
 a,b=np.asarray(a),np.asarray(b)
 return None if a.shape!=b.shape else float(np.max(np.abs(a-b)))
rows=[]
for case in ('c60_17093','tio2_anatase'):
 p=BASE/case; full=read(p/'continuous2/result.pkl'); part=read(p/'first1/result.pkl')
 a,b=full.records[0],part.records[0]
 log=[json.loads(s) for s in (p/'requests.jsonl').read_text().splitlines()]
 alog=[x for x in log if x['leg']=='continuous2'];blog=[x for x in log if x['leg']=='first1']
 first=None
 for i,(x,y) in enumerate(zip(alog,blog)):
  if x.get('energy_eV')!=y.get('energy_eV') or x.get('fmax_eV_A')!=y.get('fmax_eV_A'):
   first=dict(request=i+1,energy_a=x.get('energy_eV'),energy_b=y.get('energy_eV'),fmax_a=x.get('fmax_eV_A'),fmax_b=y.get('fmax_eV_A'));break
 stages=[]
 for i,(x,y) in enumerate(zip(a.climb,b.climb)):
  d={k:delta(x[k],y[k]) for k in ('center','direction','actual_anchor') if k in x and k in y}
  stages.append(dict(index=i,coordinate_deltas=d,requests=[x.get('requests'),y.get('requests')],true_energy=[x.get('true_energy'),y.get('true_energy')]))
 rows.append(dict(case=case,initial_position_delta=delta(full.initial.atoms.positions,part.initial.atoms.positions),initial_energy_delta=full.initial.energy-part.initial.energy,
   first_outer_requests=[a.evaluation_requests,b.evaluation_requests],first_landing_position_delta=delta(a.landing.atoms.positions,b.landing.atoms.positions),
   first_landing_energy_delta=a.landing.energy-b.landing.energy,first_ledger_difference=first,stages=stages,
   first_selection=[a.starter_selection,b.starter_selection]))
(BASE/'prefix-analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
