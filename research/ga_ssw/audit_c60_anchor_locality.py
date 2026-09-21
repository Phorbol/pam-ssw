"""No-PES audit of paper-anchor locality and atom-resolved fragment identity."""
import json
from pathlib import Path
import numpy as np
import networkx as nx
from ase import Atoms
B=Path('research/ga_ssw/evidence')
paths={'ordinary_margin01':B/'hard-c60-mace-omat-ordinary-margin01-single-step/results/ssw-seed3/result.json',
       'paper_LS':B/'hard-c60-mace-omat-single-step/whole-run/results/paper-seed3/result.json'}
rows=[]
for name,path in paths.items():
 r=json.loads(path.read_text());a=Atoms(**r['initial']['atoms']);x=a.positions;rng=np.random.default_rng(3)
 glob=rng.normal(size=x.shape)/np.sqrt(a.get_masses()[:,None]);glob/=np.linalg.norm(glob)
 eligible=[(i,j) for i in range(len(a)) for j in range(i+1,len(a)) if np.linalg.norm(x[j]-x[i])>3.]
 pair=eligible[int(rng.integers(len(eligible)))];lam=float(rng.uniform(.1,1.5))
 loc=np.zeros_like(glob);loc[pair[0]]=x[pair[1]]-x[pair[0]];loc[pair[1]]=-loc[pair[0]]
 anchor=glob+lam*loc;anchor/=np.linalg.norm(anchor)
 xx=np.asarray(r['records'][0]['landing']['atoms']['positions']);ds=np.linalg.norm(xx[:,None]-xx[None,:],axis=-1);np.fill_diagonal(ds,np.inf)
 g=nx.from_numpy_array(ds<1.6399999618530273);components=sorted((sorted(c) for c in nx.connected_components(g)),key=len)
 rows.append(dict(arm=name,source=str(path),seed=3,pair_zero_based=list(pair),mixing_coefficient=lam,
   initial_pair_distance=float(np.linalg.norm(x[pair[0]]-x[pair[1]])),
   raw_local_vector_norm=float(np.linalg.norm(loc)),
   anchor_max_error=float(np.max(abs(anchor-np.asarray(r['records'][0]['initial_direction'])))),
   pair_squared_norm_share=float(np.sum(anchor[list(pair)]**2)),
   landing_components_zero_based=components,
   landing_selected_pair_distance=float(np.linalg.norm(xx[pair[0]]-xx[pair[1]]))))
(B/'c60-anchor-locality.json').write_text(json.dumps(dict(additional_EF=0,rows=rows,
 interpretation='diagnostic of one seeded paper operator; consistent with paper eq1-2, not proof of a bug or of universal fragmentation'),indent=2)+'\n')
