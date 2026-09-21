"""Offline explicit-topology geometry audit; no calculator/PES calls."""
import argparse,json,numpy as np
from pathlib import Path
from ase import Atoms
from ase.geometry import find_mic
ROOT=Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity'); FIX=ROOT/'tests/standalone/fixtures/type2_xxxii.extxyz'; TOPO=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc')
def ad(d):return Atoms(numbers=d['numbers'],positions=d['positions'],cell=d['cell'],pbc=d['pbc'])
def pairs():return [tuple(int(z)-1 for z in q) for line in (TOPO/'blist').read_text().splitlines() if len((q:=line.split('#',1)[0].split()))==2]
def bond(a,i,j):
 raw=a.positions[j]-a.positions[i];mic,dist=find_mic(raw,a.cell.array,a.pbc);mic=np.asarray(mic); frac_delta=raw@np.linalg.inv(a.cell.array);mic_frac=mic@np.linalg.inv(a.cell.array);off=np.rint(mic_frac-frac_delta).astype(int)
 assert abs(float(np.linalg.norm(mic))-float(dist))<1e-10
 assert abs(float(np.linalg.norm((frac_delta+off)@a.cell.array))-float(dist))<1e-8
 return {'i':i,'j':j,'distance_A':float(dist),'image_offset':off.tolist()}
def metrics(a,bs):
 vals=[bond(a,i,j) for i,j in bs]; ii,jj=np.triu_indices(len(a),1); allp=find_mic(a.positions[jj]-a.positions[ii],a.cell.array,a.pbc)[1]
 return {'volume_A3':a.get_volume(),'interatomic_mic_min_A':min(allp),'bond_count':len(vals),'bond_length_min_A':min(x['distance_A'] for x in vals),'bond_length_max_A':max(x['distance_A'] for x in vals),'bond_lengths':vals}
def atomdict(v):
 if isinstance(v,dict) and 'atoms' in v:v=v['atoms']
 if isinstance(v,dict) and all(k in v for k in ('numbers','positions','cell','pbc')):
  try:return ad(v)
  except Exception:return None
 return None
def labeled(root):
 out={}
 for key in ('initial','current','best','last_work','landing'):
  z=atomdict(root.get(key)) if isinstance(root,dict) else None
  if z is not None:out[key]=z
 mins=root.get('minima',[]) if isinstance(root,dict) else []
 for i,v in enumerate(mins):
  z=atomdict(v)
  if z is not None:out[f'minima[{i}]']=z
 recs=root.get('records',[]) if isinstance(root,dict) else []
 for i,r in enumerate(recs):
  if not isinstance(r,dict):continue
  for key in ('last_work','landing'):
   z=atomdict(r.get(key))
   if z is not None:out[f'records[{i}].{key}']=z
 return out
def main(path):
 from ase.io import read
 inp=read(FIX);bs=pairs();x=json.loads(Path(path).read_text());root=x.get('result',x); structs=labeled(root)
 out={'source_input':str(FIX),'result':str(path),'topology_blist':str(TOPO/'blist'),'bond_definition':'explicit blist pairs only; no cutoff or phase identity','mic_definition':'ASE find_mic for interatomic pairs; self images are excluded from interatomic_mic_min_A','structures':{k:metrics(v,bs) for k,v in {'input':inp,**structs}.items()}}
 if 'initial' in structs:
  out['bond_delta_vs_initial']={k:[{'i':u['i'],'j':u['j'],'delta_A':v['distance_A']-u['distance_A'],'initial_image_offset':u['image_offset'],'structure_image_offset':v['image_offset']} for u,v in zip(out['structures']['initial']['bond_lengths'],m['bond_lengths'])] for k,m in out['structures'].items() if k not in ('input','initial')}
 o=Path(path).with_name('geometry-analysis.json');o.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(o),'labels':list(structs),'bond_count':len(bs)}))
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('result');main(p.parse_args().result)
