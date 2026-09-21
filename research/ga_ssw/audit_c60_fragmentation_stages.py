"""Zero-PES graph diagnostics on every saved completed C60 LS quench."""
import json
from pathlib import Path
import numpy as np
import networkx as nx
BASE=Path('research/ga_ssw/evidence/hard-c60-mace-omat-single-step/whole-run/results/paper-seed3')
quenches=json.loads((BASE/'quenches.json').read_text()); rows=[]
for index,entry in enumerate(quenches):
 x=np.asarray(entry['result']['atoms']['positions']);dist=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
 np.fill_diagonal(dist,np.inf);graphs={}
 for cutoff in (1.6399999618530273,1.7):
  g=nx.from_numpy_array(dist<cutoff)
  components=sorted((sorted(c) for c in nx.connected_components(g)),key=len,reverse=True)
  gap=min((float(dist[np.ix_(a,b)].min()) for i,a in enumerate(components) for b in components[i+1:]),default=None)
  degrees,counts=np.unique(list(dict(g.degree()).values()),return_counts=True)
  graphs[str(cutoff)]=dict(component_sizes=list(map(len,components)),degrees={str(d):int(c) for d,c in zip(degrees,counts)},intercomponent_gap=gap)
 label='initial' if index==0 else 'LS_prequench' if index==1 else 'true_landing' if index==len(quenches)-1 else f'Gaussian_{index-1}'
 rows.append(dict(label=label,completed_at_call=entry['call'],graphs=graphs))
report=dict(additional_EF=0,scope='completed quench geometries; cutoff graph is a structural diagnostic, not a chemical/pathway proof',rows=rows)
(BASE/'fragmentation-stage-audit.json').write_text(json.dumps(report,indent=2)+'\n')
