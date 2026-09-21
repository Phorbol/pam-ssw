"""Zero-EF site registry relative to ASE ideal Cu(111) adsorption-site lattice."""
import json
from pathlib import Path
import numpy as np
from ase.build import fcc111
out=Path('research/ga_ssw/evidence/constrained-cu111-emt')
r=json.loads((out/'result.json').read_text())['result'];slab=fcc111('Cu',size=(2,2,3),a=3.6,vacuum=8.)
info=slab.info['adsorbate_info'];cell=np.array(info['cell']);results=[]
for label,landing in [('initial',r['initial']),('landing',r['minima'][-1])]:
 xy=np.array(landing['atoms']['positions'])[-1,:2];distances={}
 for name,frac in info['sites'].items():
  origin=np.array(frac)@cell
  candidates=[(float(np.linalg.norm(xy-origin-np.array([i,j])@cell)),[i,j]) for i in range(-3,4) for j in range(-3,4)]
  d,image=min(candidates);distances[name]=dict(lateral_distance_A=d,image=image)
 results.append(dict(role=label,adatom_xy_A=xy.tolist(),nearest_site=min(distances,key=lambda k:distances[k]['lateral_distance_A']),distances=distances))
report=dict(scope='geometric registry relative to fixed ideal Cu(111) substrate; no Hessian, saddle, diffusion rate or general efficiency qualification',calls=0,primitive_adsorption_cell_A=cell.tolist(),sites=info['sites'],results=results)
(out/'registry.json').write_text(json.dumps(report,indent=2)+'\n');(out/'registry-script.py').write_text(Path(__file__).read_text());print(json.dumps(report,indent=2))
