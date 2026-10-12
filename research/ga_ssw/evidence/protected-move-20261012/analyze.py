"""Independent ledger/geometry readout; no PES evaluations or raw writes."""
import json,sys
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
run,out=map(Path,sys.argv[1:3])
previous=Path(sys.argv[3]) if len(sys.argv)>3 else None
p=json.loads((run/'provenance.json').read_text()); rows=[]
for row in p['runs']:
 folder=run/(row['case']+'-'+row['arm'])
 result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
 counts={}
 for name,key in [('requests.jsonl','search_requests'),('fresh-requests.jsonl','fresh_requests')]:
  records=[json.loads(s) for s in (folder/name).read_text().splitlines()] if (folder/name).exists() else []
  counts[key]=sum(r['kind'] in ('search','search_failure') for r in records)
  assert counts[key]==row.get(key,0),(folder,key,counts,row)
 stages=[s for rec in result['records'] for s in rec['climb']] if result else []
 moves=[s.get('protected_move',s.get('move_diagnostics')) for s in stages]
 moves=[m for m in moves if m is not None]
 r={k:row.get(k) for k in ['case','arm','status','failure_class','fresh_failure_class','search_requests','search_calculate_calls','fresh_requests','fresh_calculate_calls','search_boundary','fresh_boundary','search_elapsed_seconds','arm_elapsed_seconds','fresh_endpoint']}
 r.update(constructed_gaussians=sum('width' in s for s in stages),guard_stages=len(moves),rejected_trials=sum(not a['accepted'] for m in moves for a in m['attempts']),shortened_moves=sum(m.get('actual_width',0)>0 and m['actual_width']<m['requested_width']-1e-12 for m in moves),release_reasons=[rec['status'] for rec in result['records']] if result else [],widths=[s['width'] for s in stages if 'width' in s])
 if result:
  r['initial_energy_eV']=result['initial']['energy']
  landings=[rec['landing'] for rec in result['records'] if rec.get('landing') is not None]
  r['landing_delta_eV']=[l['energy']-r['initial_energy_eV'] for l in landings]
  fresh=row.get('fresh_endpoint',{})
  if landings and 'energy_eV' in fresh:
   r['fresh_energy_error_eV']=fresh['energy_eV']-landings[-1]['energy']
 if (folder/'endpoint-before-fresh.extxyz').exists():
  a=read(folder/'endpoint-before-fresh.extxyz')
  if not a.pbc.any():
   d=a.get_all_distances(); geometry=[]
   for cutoff in (1.64,1.7,1.8):
    todo=set(range(len(a))); components=[]
    while todo:
     stack=[todo.pop()];g=[]
     while stack:
      i=stack.pop();g.append(i);new={j for j in todo if d[i,j]<cutoff};todo-=new;stack.extend(new)
     components.append(g)
    adjacency=(d<cutoff)&~np.eye(len(a),dtype=bool)
    deg,cnt=np.unique(adjacency.sum(axis=1),return_counts=True)
    geometry.append(dict(cutoff_A=cutoff,component_sizes=sorted(map(len,components)),degrees=dict(zip(map(str,deg.tolist()),cnt.tolist()))))
   r['cluster_geometry']=geometry
  else:
   i,j,d=neighbor_list('ijd',a,3.,self_interaction=False)
   r['periodic_3A_image_counts']=np.bincount(i,minlength=len(a)).tolist()
   r['nearest_periodic_distance_A']=float(min(d))
 if row['arm']=='protected-off' and result and previous is not None:
  old=json.loads((previous/(row['case']+'-S2')/'first-segment-result.json').read_text())
  r['prior_off_search_requests']=old['evaluation_requests']
  r['matches_prior_off_requests']=old['evaluation_requests']==result['evaluation_requests']
  old_l=old['records'][0]['landing']; new_l=result['records'][0]['landing']
  if old_l and new_l:
   r['prior_off_landing_max_position_difference_A']=float(np.max(np.abs(np.asarray(old_l['atoms']['positions'])-np.asarray(new_l['atoms']['positions']))))
   r['prior_off_landing_energy_difference_eV']=new_l['energy']-old_l['energy']
 rows.append(r)
pairs=[]
for case in sorted({r['case'] for r in rows}):
 if not all((run/(case+'-'+arm)/'result.json').exists() for arm in ('protected-off','protected-on')):
  pairs.append(dict(case=case,prefix_unavailable='one or both result files unavailable'))
  continue
 off=json.loads((run/(case+'-protected-off')/'result.json').read_text())
 on=json.loads((run/(case+'-protected-on')/'result.json').read_text())
 before=[]
 off_climb=off['records'][0]['climb'] if off['records'] else []
 on_climb=on['records'][0]['climb'] if on['records'] else []
 for a,b in zip(off_climb,on_climb):
  if 'center' not in a or 'center' not in b: break
  m=b.get('protected_move',b.get('move_diagnostics',{}))
  triggered=any(not t['accepted'] for t in m.get('attempts',[]))
  before.append(dict(stage=b['index'],first_guard_rejection=triggered,center_max_difference_A=float(np.max(np.abs(np.asarray(a['center'])-np.asarray(b['center'])))),CBD_direction_max_difference=float(np.max(np.abs(np.asarray(a['direction'])-np.asarray(b.get('requested_direction',b.get('direction'))))))))
  if triggered: break
 pairs.append(dict(case=case,prefix_through_first_guard=before))
output=dict(raw_run=str(run),scope='development paired single outer step; no general efficiency claim',search_requests=sum(r['search_requests'] for r in rows),fresh_requests=sum(r['fresh_requests'] or 0 for r in rows),rows=rows,pair_prefix=pairs)
assert output['search_requests']==p['search_requests_total']
assert output['fresh_requests']==p['fresh_requests_total']
out.mkdir(exist_ok=False); (out/'analysis.json').write_text(json.dumps(output,indent=2))
for r in rows: print(json.dumps(r))
