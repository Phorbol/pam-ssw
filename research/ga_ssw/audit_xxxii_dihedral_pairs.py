"""Read-only actual topology audit for native DihedralAmber conversion."""
import json,collections
from pathlib import Path
import numpy as np
from ase.io import read
p=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc/lmp.data')
sections={};current=None
for line in p.read_text().splitlines():
 line=line.split('#')[0].strip()
 if not line:continue
 if line[0].isalpha():current=line;sections[current]=[]
 elif current:sections[current].append(line.split())
coeff={int(r[0]):list(map(float,r[1:])) for r in sections['Dihedral Coeffs']}
bonds=[(int(r[2])-1,int(r[3])-1) for r in sections['Bonds']];neighbors=collections.defaultdict(set)
for i,j in bonds:neighbors[i].add(j);neighbors[j].add(i)
def distance(i,j):
 seen={i};front={i}
 for n in range(1,5):
  front={k for v in front for k in neighbors[v]}-seen
  if j in front:return n
  seen|=front
 return None
pairs=collections.defaultdict(list)
for r in sections['Dihedrals']:
 ident,typ,*indices=map(int,r);ends=tuple(sorted((indices[0]-1,indices[3]-1)));K,n,phase,wlj,wc=coeff[typ]
 if .83<wc<.84:wc=5/6
 pairs[ends].append(dict(id=ident,type=typ,weight_lj=wlj,weight_coul=wc))
a=read('tests/standalone/fixtures/type2_xxxii.extxyz');rows=[]
for (i,j),ds in sorted(pairs.items()):
 rows.append(dict(pair=[i,j],bond_graph_distance=distance(i,j),dihedrals=ds,weight_lj=sum(x['weight_lj'] for x in ds),weight_coul=sum(x['weight_coul'] for x in ds),nonzero_count=sum(x['weight_lj']!=0 or x['weight_coul']!=0 for x in ds),distance_A=float(np.linalg.norm(a.positions[i]-a.positions[j]))))
all_three={(i,j) for i in range(len(a)) for j in range(i+1,len(a)) if distance(i,j)==3}
summary=dict(dihedral_count=sum(map(len,pairs.values())),unique_endpoint_pairs=len(rows),graph_distance_counts=dict(collections.Counter(str(r['bond_graph_distance']) for r in rows)),total_weight_counts=dict(collections.Counter(str((r['weight_lj'],r['weight_coul'])) for r in rows)),nonzero_count_counts=dict(collections.Counter(r['nonzero_count'] for r in rows)),graph_three_pairs=len(all_three),graph_three_without_dihedrals=sorted(all_three-set(pairs)),nonzero_weight_shorter_graph_pairs=[r['pair'] for r in rows if r['bond_graph_distance']!=3 and r['nonzero_count']],max_distance_A=max(r['distance_A'] for r in rows),source=str(p),PES_requests=0)
Path('research/ga_ssw/evidence/native-dihedral-amber/pair-audit.json').write_text(json.dumps(dict(summary=summary,pairs=rows),indent=2)+'\n');print(json.dumps(summary,indent=2))
