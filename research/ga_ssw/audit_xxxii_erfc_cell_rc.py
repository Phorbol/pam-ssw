"""Offline projection of known real-space erfc force error, including cell/RC.

No calculator. Fixed screening G; does not explain adaptive G or reciprocal error.
"""
from pathlib import Path
import json,re,hashlib
import numpy as np
from ase.io import read
from ase import units
from ase.neighborlist import neighbor_list
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research/ga_ssw/evidence/xxxii-lammps-qualification-table0-ewald12'
OUT=ROOT/'research/ga_ssw/evidence/xxxii-erfc-cell-rc-audit';OUT.mkdir(exist_ok=True)
p=json.loads((SRC/'plan.json').read_text());old=json.loads((SRC/'result.json').read_text())
a=read(SRC/'type2_xxxii.extxyz');top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=len(a));a=unwrap_rigid_molecules(a,top.bonds).atoms
lines=(SRC/'lmp.data').read_text().splitlines();start=lines.index(' Atoms');data=[]
for line in lines[start+1:]:
 if not line.strip():
  if data:break
  continue
 t=line.split()
 if len(t)>=7 and t[0].isdigit():data.append((int(t[0]),float(t[3])))
assert sorted(i for i,_ in data)==list(range(1,len(a)+1));charge=np.array([v for _,v in sorted(data)])
G=float(re.findall(r'G vector \(1/distance\) = ([0-9.eE+-]+)',(SRC/'engine-0.log').read_text())[0])
i,j,dr,r=neighbor_list('ijDd',a,10.,self_interaction=False);dr=-dr
x=G*r;t=1/(1+.3275911*x);aa=np.array([.254829592,-.284496736,1.421413741,-1.453152027,1.061405429])
poly=aa[0]+t*(aa[1]+t*(aa[2]+t*(aa[3]+t*aa[4])))
dpoly=aa[1]+t*(2*aa[2]+t*(3*aa[3]+t*4*aa[4]));ep=np.exp(-x*x)
dA=ep*(-.3275911*t*t*(poly+t*dpoly)-2*x*t*poly)
mag=332.06371*(units.kcal/units.mol)*charge[i]*charge[j]/r**2*x*(dA+2/np.sqrt(np.pi)*ep)
v=mag[:,None]*dr/r[:,None]
df=np.zeros_like(a.positions);np.add.at(df,i,.5*v);np.add.at(df,j,-.5*v)
virial=.5*np.einsum('ni,nj->ij',dr,v);ds=-virial/a.get_volume()
vc=SymmetricLogStrainChart(a,strain_length=p['strain_length']);q=vc.pack(a)
rc=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=p['rotation_length'],torsion_length=p['torsion_length'],strain_length=p['strain_length'])
# evaluate is only a linear coordinate pullback of supplied errors; no oracle.
vc_error=vc.evaluate(q,lambda b:(0.,df,ds)).gradient
rc_error=rc.evaluate(np.zeros(rc.dimension),lambda b:(0.,df,ds)).gradient
pred={'atomic':float(np.sum(df*np.array(p['atomic_direction']))),'RC':-float(rc_error@np.array(p['RC_direction']))}
pred.update({f'cell{k}':-float(vc_error[-6+k]) for k in range(6)})
rows=[]
for d in old['derivatives']:
 observed=d['finite_difference']-d['analytic'];expected=pred[d['direction']]
 rows.append(dict(direction=d['direction'],h=d['h'],observed_FD_minus_analytic=observed,predicted_fixedG_realspace_error=expected,remaining=observed-expected))
result=dict(PES_calls=0,G=G,ordered_image_pairs=len(r),ordered_self_images=int(np.sum(i==j)),force_error_max=float(np.linalg.norm(df,axis=1).max()),stress_error=ds.tolist(),predictions=pred,comparison=rows,sign='deltaF=F_code-F_energy, deltaStress=stress_code-stress_energy. FD-analytic=-deltaGradient; atomic therefore +sum(deltaF*u).',limits='fixed G at first logged value; finite pair polynomial mismatch only. Does not include G adaptation, reciprocal truncation, runtime pair rounding, or full native GAFF agreement.',inputs={name:hashlib.sha256((SRC/name).read_bytes()).hexdigest() for name in ['plan.json','result.json','type2_xxxii.extxyz','lmp.data','rigidbody','blist']})
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,comparisons=rows),indent=2))
