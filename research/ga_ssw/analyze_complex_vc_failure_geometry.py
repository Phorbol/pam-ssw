"""Zero-PES failure and coordination audit; frozen results read only."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
ROOT=Path('research/ga_ssw/prospective/complex-vc-feasibility')
OUT=Path('research/ga_ssw/evidence/complex-vc-failure-geometry');OUT.mkdir(exist_ok=True)
def atom(d):return Atoms(**{k:d[k] for k in ('numbers','positions','cell','pbc')})
def geometry(a):
 d=a.get_all_distances(mic=True);np.fill_diagonal(d,np.inf)
 z=a.numbers;rows=[]
 for i in np.flatnonzero(np.isin(z,[1,13])):
  indices=np.flatnonzero(z==8);order=indices[np.argsort(d[i,indices])]
  rows.append(dict(atom=int(i),species=a[i].symbol,nearest_oxygen=[dict(atom=int(j),distance=float(d[i,j])) for j in order[:7]],counts={str(c):int(np.sum(d[i,indices]<c)) for c in ([1.1,1.2,1.3] if z[i]==1 else [2.,2.2,2.4])}))
 return dict(volume=float(a.get_volume()),minimum_distance=float(d.min()),cell_singular_values=np.linalg.svd(a.cell.array,compute_uv=False).tolist(),coordination=rows)
audit=json.load(open(ROOT/'offline-analysis/analysis.json'));out=dict(extra_PES_requests=0,planned_runs=8,runs=[],note='Distance cutoffs are descriptive sweeps, not bond orders; atoms retain original indices.')
for row in audit['runs']:
 r=json.load(open(row['result_path']));entry={k:row[k] for k in ('name','requests','initial_requests','wall_seconds','outcome_counts')};entry['stages']=[dict(index=e.get('index',-1),status=e.get('status'),requests=e['requests'],accepted=e.get('accepted'),gaussians=[{k:c[k] for k in ('index','status','weight','objective','error') if k in c} for c in e.get('climb',[])]) for e in r['records']]
 entry['landings']=[dict(index=x['index'],accepted=x['accepted'],objective=x['objective'],certificate=x['certificate'],geometry=geometry(atom(x['atoms']))) for x in r['landings']]
 if row['name']=='brookite48-joint-seed17':
  chart=SymmetricLogStrainChart(atom(r['common_start']),strain_length=r['joint_config']['strain_length']);entry['biased_geometry']=[]
  for stage in r['records'][1]['climb']:
   if 'q' in stage:
    a=chart.unpack(np.array(stage['q']));np.testing.assert_allclose(a.cell.array,stage['cell'],atol=1e-12)
    entry['biased_geometry'].append(dict(index=stage['index'],status=stage['status'],**geometry(a)))
 out['runs'].append(entry)
(OUT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out['runs']:
 print(r['name'],r['requests'],r['initial_requests'],r['outcome_counts'])
 if r['name']=='aloh26-joint-seed17':
  initial,final=r['landings'];print('AlOH volume',initial['geometry']['volume'],final['geometry']['volume'])
  for before,after in zip(initial['geometry']['coordination'],final['geometry']['coordination']):
   print(before['species'],before['atom'],before['counts'],after['counts'],'nearest',before['nearest_oxygen'][:2],after['nearest_oxygen'][:2])
 if 'biased_geometry' in r:print('brookite volumes',[round(g['volume'],3) for g in r['biased_geometry']],'minimum distances',[round(g['minimum_distance'],3) for g in r['biased_geometry']])
