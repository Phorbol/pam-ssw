"""Zero-PES algebraic scale audit; not a native direction or search reproduction."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
source=Path('research/ga_ssw/evidence/hard-c60-mace-omat-single-step/whole-run/results/paper-seed3/result.json')
r=json.loads(source.read_text());a=Atoms(**r['initial']['atoms']);rng=np.random.default_rng(3)
g=rng.normal(size=(len(a),3))/np.sqrt(a.get_masses()[:,None]);g/=np.linalg.norm(g)
pairs=[(i,j) for i in range(len(a)) for j in range(i+1,len(a)) if np.linalg.norm(a.positions[j]-a.positions[i])>3.]
i,j=pairs[int(rng.integers(len(pairs)))];lam=float(rng.uniform(.1,1.5))
l=np.zeros_like(g);l[i]=a.positions[j]-a.positions[i];l[j]=-l[i]
rows=[]
for name,local in [('paper_raw',l),('unit_local_only',l/np.linalg.norm(l))]:
 v=g+lam*local;v/=np.linalg.norm(v)
 rows.append(dict(mode=name,local_norm=float(np.linalg.norm(local)),coefficient=lam,weighted_local_norm=float(lam*np.linalg.norm(local)),selected_pair_squared_share=float((v[[i,j]]**2).sum())))
assert np.max(np.abs((g+lam*l)/np.linalg.norm(g+lam*l)-np.asarray(r['records'][0]['initial_direction'])))<1e-14
out=dict(source=str(source),additional_EF=0,pair_zero_based=[i,j],rows=rows,scope='One seed; only independent local normalization changed algebraically. Omits native neighbor additions, selection, constraints, random generator and Ratio_Local. No native parity or search efficacy claim.')
Path('research/ga_ssw/evidence/c60-normalized-local-scale.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
