import json,itertools
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment
b=Path('research/ga_ssw/evidence');q=b/'type4-direction-strict-qualification'
a={k:Atoms(**json.loads((q/k/'strict-atoms.json').read_text())) for k in ('initial','physical_mask_landing','source_mask_landing')}
d=json.loads((b/'type4-initial-negative-mode/result.json').read_text())
a.update({k:Atoms(**v['atoms']) for k,v in d['arms'].items()})
results={}
for ka,kb in [('physical_mask_landing','source_mask_landing'),('unperturbed','negative'),('unperturbed','positive'),('unperturbed','physical_mask_landing'),('unperturbed','source_mask_landing')]:
 x,y=a[ka],a[kb];item={}
 for label,indices in [('mobile',np.arange(297,514)),('adsorbate',np.arange(486,514)),('mobile_support',np.arange(297,486))]:
  distances=[]
  for z in np.unique(x.numbers[indices]):
   ix=indices[x.numbers[indices]==z];iy=indices[y.numbers[indices]==z]
   delta=x.positions[ix,None,:]-y.positions[None,iy,:]
   _,cost=find_mic(delta.reshape(-1,3),x.cell,x.pbc);cost=cost.reshape(len(ix),len(iy));r,c=linear_sum_assignment(cost**2);distances.extend(cost[r,c])
  item[label]=dict(rms=float(np.sqrt(np.mean(np.square(distances)))),max=float(max(distances)))
 results[ka+'__'+kb]=item
results['scope']='element-preserving periodic Hungarian assignment with fixed support registration; no rotation/alignment or kinetic-connectivity inference; no new PES calls'
(q/'structure-comparison.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
