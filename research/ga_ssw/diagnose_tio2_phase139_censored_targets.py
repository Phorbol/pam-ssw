"""Two fresh EFS checks of frozen phase139 biased targets; no continuation."""
import json,hashlib,os,sys,time,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart, ASEStressSurface

ROOT=Path(__file__).resolve().parents[2]
EVID=ROOT/'research/ga_ssw/evidence'
OUT=EVID/'tio2-phase139-frozen-endpoint-diagnostic'
SOURCES=[EVID/'tio2-phase139-joint-memory-compare/memory400/result.json',EVID/'tio2-phase139-joint-central-ritz-memory400/run-central-ritz-seed3/result.json']
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(MODEL)=='0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5'
plan=dict(max_EFS=2,source_results=[dict(path=str(p),sha256=sha(p)) for p in SOURCES],model=str(MODEL),model_sha256=sha(MODEL),scope='independent last accepted biased-target gradient; zero optimization steps, no continuation or new landing')
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');shutil.copy2(__file__,OUT/'script.py')
import torch,ase
from mace.calculators import MACECalculator
torch.set_num_threads(1);torch.set_num_interop_threads(1)
(OUT/'runtime-environment.json').write_text(json.dumps(dict(python=sys.executable,numpy=np.__version__,ase=ase.__version__,torch=torch.__version__,threads=torch.get_num_threads(),environment={k:os.environ.get(k) for k in ('PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),indent=2)+'\n')
rows=[];start=time.monotonic()
for p in SOURCES:
 r=json.loads(p.read_text());event=r['records'][1];cfg=r['joint_config'];a=Atoms(**event['chart_reference']);chart=SymmetricLogStrainChart(a,strain_length=cfg['strain_length']);q=np.array(event['climb'][-1]['q'])
 calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64');s=ASEStressSurface(calc)
 ev=chart.evaluate(q,s.evaluate,pressure=cfg['pressure']);g=ev.gradient.copy();energy=ev.objective
 for term in event['frozen_gaussians']:
  n=np.array(term['direction']);z=float((q-term['center'])@n);width=term['width'];v=term['weight']*np.exp(-.5*(z/width)**2);energy+=v;g-=v*z/width**2*n
 rows.append(dict(source=str(p),EFS=s.requests,index=event['climb'][-1]['index'],nGaussians=len(event['frozen_gaussians']),true_E=ev.energy,biased_objective=float(energy),biased_gradient_norm=float(np.linalg.norm(g)),gradient_tolerance=cfg['gradient_tol'],biased_stationarity_pass=bool(np.linalg.norm(g)<=cfg['gradient_tol']),atomic_gradient_norm=float(np.linalg.norm(g[:-6])),strain_gradient_norm=float(np.linalg.norm(g[-6:])),true_fmax=float(np.linalg.norm(ev.forces,axis=1).max()),true_stress_max=float(abs(ev.stress).max()),volume=ev.volume,q=q.tolist(),biased_gradient=g.tolist()))
 (OUT/'result.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n')
report=dict(status='completed',total_EFS=sum(x['EFS'] for x in rows),wall_seconds=time.monotonic()-start,rows=rows)
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({**report,'rows':[{k:v for k,v in row.items() if k not in ('q','biased_gradient')} for row in rows]},indent=2))
