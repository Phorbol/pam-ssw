from itertools import product
from pathlib import Path
import json
import re
import numpy as np
from ase.io import read
from ase import units

P=0.3275911
A=np.array([0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429])
F=2/np.sqrt(np.pi); C=332.06371*(units.kcal/units.mol); RC=10.
ROOT=Path(__file__).resolve().parent/'evidence/xxxii-lammps-qualification-table0-ewald12'
OUT=Path(__file__).resolve().parent/'evidence/xxxii-erfc-floor-audit'

def charges(path):
    lines=path.read_text().splitlines(); k=lines.index(' Atoms'); out=[]
    for line in lines[k+1:]:
        if not line.strip():
            if out: break
            continue
        z=line.split()
        if len(z)>=7 and z[0].isdigit(): out.append((int(z[0]),float(z[3])))
    if len(out)!=172 or sorted(i for i,_ in out)!=list(range(1,173)):
        raise RuntimeError('atom IDs are not exactly 1..172')
    return np.array([charge for _,charge in sorted(out)])

def measured_g(path):
    values=[float(x) for x in re.findall(r'G vector \(1/distance\) = ([0-9.eE+-]+)', path.read_text())]
    if not values or not np.all(np.isfinite(values)):
        raise RuntimeError('no finite G vector found in saved engine log')
    return values[0], values

def mismatch(r):
    x=G*r; t=1/(1+P*x); q=A[0]+t*(A[1]+t*(A[2]+t*(A[3]+t*A[4])))
    poly=t*q; ep=np.exp(-x*x); dq=A[1]+t*(2*A[2]+t*(3*A[3]+t*4*A[4]))
    dtdx=-P*t*t; dA=ep*(dtdx*(q+t*dq)-2*x*poly)
    return x*(dA+F*ep)

atoms=read(ROOT/'type2_xxxii.extxyz'); q=charges(ROOT/'lmp.data'); cell=atoms.cell.array; pos=atoms.positions
G,G_values=measured_g(ROOT/'engine-0.log')
invcell=np.linalg.inv(cell)
df=np.zeros_like(pos); n=0; asum=0.
for i in range(172):
  for j in range(i+1,172):
    frac=(pos[i]-pos[j])@invcell
    bounds=RC*np.linalg.norm(invcell,axis=0)
    ranges=[range(int(np.floor(frac[k]-bounds[k]))-1,int(np.ceil(frac[k]+bounds[k]))+2) for k in range(3)]
    for sh in product(*ranges):
      dr=pos[i]-(pos[j]+np.asarray(sh)@cell); r=np.linalg.norm(dr)
      if 0<r<RC:
        mag=C*q[i]*q[j]/r**2*mismatch(r); v=mag*dr/r; df[i]+=v; df[j]-=v; n+=1; asum+=abs(mag)
u=np.asarray(json.loads((ROOT/'plan.json').read_text())['atomic_direction'])
proj=float(np.sum(df*u))
result={'G':G,'all_saved_G_values':G_values,'pairs':n,'sum_abs_pair_error':asum,
        'force_error_max':float(np.linalg.norm(df,axis=1).max()),
        'force_error_rms':float(np.sqrt(np.mean(np.sum(df*df,axis=1)))),
        'atomic_projection_code_minus_energy':proj,
        'predicted_FD_minus_analytic':-proj,
        'image_bound':'per-pair frac displacement plus RC*norm(inv(cell) column), with one-cell safety padding',
        'self_image_pairs':'omitted: their atomic-displacement contribution is identically zero'}
OUT.mkdir(exist_ok=True)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
