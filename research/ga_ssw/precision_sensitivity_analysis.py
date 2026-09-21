"""Derive fixed-direction curvature/residual from saved precision rows only."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'research/ga_ssw/evidence/hard-c60-gfn2-paper-ls-memory400-single-step'
P=B/'precision-sensitivity-v1'; F=json.load(open(B/'ritz-results-v2/frozen-objective-v2.json'))
rows=[json.loads(x) for x in open(P/'evaluations.jsonl')]
rows += [json.loads(x) for x in open(P/'continuation-v1/evaluations.jsonl')]
by={(float(x['accuracy']),x['point']):x for x in rows}
anchor=np.asarray(F['projected_normalized_anchor']); old=json.load(open(B/'results/paper-seed3/offline-stage10/summary.json'))
dirs={'dimer':np.asarray(old['direction']),'ritz':np.asarray(json.load(open(B/'ritz-results-v2/result.json'))['direction'])}
h=1e-4; layers=[0.001,0.0001,0.00001]; labels={0.001:'0.001',0.0001:'0.0001',0.00001:'0.00001'}; out={'accuracy_layers_present':[labels[x] for x in layers],'accuracy_layers_pending':[],'points':{},'adjacent_projected_force_delta_over_h':{},'source':'nine saved precision rows in two non-continuous segments','extra_PES':0}
for layer in layers:
 f0=np.asarray(by[(layer,'center')]['projected_forces'])
 out['points'][labels[layer]]={}
 for name,n in dirs.items():
  n=n.ravel(); hn=((f0-np.asarray(by[(layer,name)]['projected_forces']))/h).ravel()-100*np.dot(anchor.ravel(),n)*anchor.ravel(); c=float(n@hn)
  out['points'][labels[layer]][name]={'curvature':c,'residual':float(np.linalg.norm(hn-c*n)),'force_requests':3,'fixed_direction':True,'below_tol_0.02':bool(np.linalg.norm(hn-c*n)<.02)}
for hi,lo in zip(layers,layers[1:]):
 out['adjacent_projected_force_delta_over_h'][f'{labels[hi]}->{labels[lo]}']={}
 for point in ('center','dimer','ritz'):
  delta=(np.asarray(by[(lo,point)]['projected_forces'])-np.asarray(by[(hi,point)]['projected_forces']))/h
  out['adjacent_projected_force_delta_over_h'][f'{labels[hi]}->{labels[lo]}'][point]={'norm':float(np.linalg.norm(delta)),'max_component':float(np.max(np.abs(delta)))}
json.dump(out,open(P/'results-derived.json','w'),indent=2); print(json.dumps(out,indent=2))
