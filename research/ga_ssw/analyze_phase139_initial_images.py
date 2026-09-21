"""Offline periodic Ti-O image neighbor audit for phase139 memory arms."""
import json,numpy as np
from pathlib import Path
from ase import Atoms
ROOT=Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity'); B=ROOT/'research/ga_ssw/evidence/tio2-phase139-joint-memory-compare'
def A(d):return Atoms(numbers=d['numbers'],positions=d['positions'],cell=d['cell'],pbc=d['pbc'])
def calc(a):
 f=a.get_scaled_positions(wrap=False); cell=np.asarray(a.cell.array); out=[]
 for i in np.where(a.numbers==22)[0]:
  rows=[]
  for j in np.where(a.numbers==8)[0]:
   for sh in np.ndindex(3,3,3):
    off=np.array(sh)-1; dr=(f[j]+off-f[i])@cell;rows.append((float(np.linalg.norm(dr)),int(j),off.tolist()))
  rows.sort();out.append({'ti_atom':int(i),'nearest6_image_resolved':[{'distance_A':d,'o_atom':j,'image_offset':o} for d,j,o in rows[:6]]})
 return out
res={'source_input':'literature/benchmark-sources/coordinates/phase-139.extxyz','arms':{}}
for arm in ['memory10','memory400']:
 x=json.load(open(B/arm/'result.json'));res['arms'][arm]={'initial_requests':x['initial_requests'],'initial_certificate':x['records'][0]['certificate'],'metrics':calc(A(x['landings'][0]['atoms']))}
(B/'initial-ti-o-image-neighbors.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v['initial_requests'] for k,v in res['arms'].items()}))
