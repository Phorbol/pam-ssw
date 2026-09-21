"""Offline graph/reference diagnostics for native-vapor oracle outputs; no PES."""
import argparse,json
from pathlib import Path
import numpy as np

def diag(pos,cri):
 x=np.asarray(pos,float); n=len(x); d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
 parent=list(range(n))
 def find(a):
  while parent[a]!=a: parent[a]=parent[parent[a]];a=parent[a]
  return a
 for i in range(n):
  for j in range(i):
   if d[i,j] < cri:
    a,b=find(i),find(j)
    if a!=b:parent[a]=b
 comp={}
 for i in range(n):comp.setdefault(find(i),[]).append(i)
 groups=list(comp.values()); cross=[]
 for a in range(len(groups)):
  for b in range(a):cross.append(float(d[np.ix_(groups[a],groups[b])].min()))
 return {'n':n,'components':[len(g) for g in groups],'component_members':groups,'min_cross_distance':min(cross) if cross else 0.0,'max_cross_distance':max(cross) if cross else 0.0}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('result');ap.add_argument('--output',required=True);a=ap.parse_args();d=json.loads(Path(a.result).read_text()); rows=[]
 for r in d['rows']:
  if r['status']!='ok': rows.append({'n':r['n'],'mode':r['mode'],'status':r['status']});continue
  before=np.asarray(r['positions']); # v1 stores post positions; synthetic cases are known from source not retained
  # Reconstruct only diagnostics where mode pair is available: compare mode 0 row in same case.
  base=next((q for q in d['rows'] if q.get('case_index')==r.get('case_index') and q.get('mode')==0 and q.get('status')=='ok'),None)
  if base is None: rows.append({'n':r['n'],'mode':r['mode'],'status':'missing_mode0'});continue
  x0=np.asarray(base['positions']); rows.append({'n':r['n'],'mode':r['mode'],'return':r['return'],'max_coord_delta':r['max_coord_delta'],'before_graph':diag(x0, r['cri']),'after_graph':diag(before,r['cri']),'all_pair_distance_max_delta':float(np.max(np.abs(np.linalg.norm(x0[:,None,:]-x0[None,:,:],axis=2)-np.linalg.norm(before[:,None,:]-before[None,:,:],axis=2))) if len(x0)>1 else 0.0), 'within_component_distance_max_delta': float(max((np.max(np.abs(np.linalg.norm(x0[g][:,None,:]-x0[g][None,:,:],axis=2)-np.linalg.norm(before[g][:,None,:]-before[g][None,:,:],axis=2))) for g in diag(x0,r['cri'])['component_members']), default=0.0))})
 Path(a.output).write_text(json.dumps({'source':a.result,'scope':'offline NumPy diagnostics, not native proof','rows':rows},indent=2)+'\n')
 print(json.dumps({'rows':len(rows)},separators=(',',':')))
if __name__=='__main__':main()
