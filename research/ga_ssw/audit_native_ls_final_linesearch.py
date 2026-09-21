import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.constraints import Hookean
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.relax import _lbfgs_inverse_product
p=Path('research/ga_ssw/evidence/constrained-gaussian-reference-20260912/cu111-ls_native')
r=json.loads((p/'result.json').read_text());event=r['records'][1];pre=event['ls_preparation'];o=pre['optimizer'];t=o['trace'];soft=FrozenBondSoftening(**pre['softening'])
ledger=[json.loads(l) for l in (p/'evaluations.jsonl').read_text().splitlines()]
active=np.arange(8,13);h=[]
for u,v in zip(t,t[1:]):
 s=np.array(v['q'])-u['q'];y=np.array(v['gradient'])-u['gradient'];h.append((s,y,1/float(s@y)))
h=h[-10:];g=np.array(o['gradient']);d=-_lbfgs_inverse_product(g,h);dn=np.linalg.norm(d.reshape(-1,3),axis=1).max()
if dn>.2:d*=.2/dn
base=np.array(t[-1]['q']);benergy=o['energy'];deriv=float(g@d)
constraints=[Hookean(**c['kwargs']) for c in r['initial']['atoms']['constraints'] if c['name']=='Hookean']
def unpack(entry):
 a=Atoms(**{k:entry['atoms'][k] for k in ('numbers','positions','cell','pbc')});e,f=soft.evaluate(a)
 he=sum(c.adjust_potential_energy(a) for c in constraints)
 return a,entry['energy']+e+he,f
out=[]
# initial quench6 + physical response-before1 precede optimizer request1.
for i,entry in enumerate(ledger[132:]):
 a,e,f=unpack(entry);q=(a.positions[active]-np.array(r['initial']['atoms']['positions'])[active]).ravel()
 alpha=2.**(-i);err=np.max(np.abs(q-(base+alpha*d)))
 out.append(dict(request=entry['request'],alpha=alpha,coordinate_error=err,
                 energy_difference=e-benergy,armijo_required_difference=1e-4*alpha*deriv))
prev=unpack(ledger[131]);j=unpack(ledger[132])
report=dict(last_accepted_request=132,steps=o['steps'],last_objective=benergy,
  final_gradient_norm=t[-1]['gradient_norm'],direction_max_atom_norm=dn,g_dot_direction=deriv,
  final_failed_trials=out,ls_force_jump_first_trial=float(np.linalg.norm(j[2]-prev[2])),
  coordinate_jump_first_trial=float(np.linalg.norm(j[0].positions-prev[0].positions)),
  physical_force_jump_first_trial=float(np.linalg.norm(np.array(ledger[132]['forces'])-ledger[131]['forces'])),
  new_oracle_requests=0)
(p/'final-line-search-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
from ase.geometry import find_mic
last=unpack(ledger[-1]);shifts=[]
for i,jj in soft.pairs:
 a0=prev[0];a1=last[0]
 v0=a0.positions[jj]-a0.positions[i];v1=a1.positions[jj]-a1.positions[i]
 m0,_=find_mic(v0,a0.cell,a0.pbc);m1,_=find_mic(v1,a1.cell,a1.pbc)
 s0=np.rint(np.linalg.solve(a0.cell.array.T,m0-v0)).astype(int);s1=np.rint(np.linalg.solve(a1.cell.array.T,m1-v1)).astype(int)
 if not np.array_equal(s0,s1):shifts.append(dict(pair=[i,jj],old=s0.tolist(),new=s1.tolist()))
report.update(last_trial_ls_force_jump=float(np.linalg.norm(last[2]-prev[2])),
 last_trial_physical_force_jump=float(np.linalg.norm(np.array(ledger[-1]['forces'])-ledger[131]['forces'])),
 last_trial_coordinate_jump=float(np.linalg.norm(last[0].positions-prev[0].positions)),
 last_trial_image_switches=shifts)
(p/'final-line-search-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('FINAL',json.dumps({k:v for k,v in report.items() if k.startswith('last_trial')}))
