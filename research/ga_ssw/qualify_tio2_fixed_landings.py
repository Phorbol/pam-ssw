"""Bounded post-hoc fixed-cell Hessian audit, no relaxation or search tuning."""
import json, os, time, hashlib, shutil
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator
from pamssw.standalone.surface import ASESurface
ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/'research/ga_ssw/evidence/tio2-fixed-ritz-staged-holdout-20260912'
OUT=ROOT/'research/ga_ssw/evidence/tio2-fixed-landing-hessian-20260912'
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
CASES=[('anatase-seed11-fixed_ritz_a100',2),('rutile-seed29-fixed_ritz_a100',2)]
def main():
 OUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUT/'runner.py')
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 manifest=dict(status='running',cases=CASES,model=str(MODEL),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),device='cpu',finite_difference_step_A=.001,planned_requests=146,scope='highest-energy force-certified fixed-Ritz landing per TiO2 input (selection restricted to fixed-Ritz baseline arms); post-hoc curvature qualification, no relaxation/parameter selection',environment={k:os.getenv(k) for k in ('PYTHONNOUSERSITE','PYTHONPATH','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES')})
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 surface=ASESurface(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False));results=[];start=time.monotonic()
 for arm,index in CASES:
  source=CAMPAIGN/arm/'result.json';d=json.loads(source.read_text());m=d['minima'][index];a=Atoms(**m['atoms']);n=len(a);h=.001;before=surface.requests
  row=dict(arm=arm,index=index,input=m,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
  try:
   e,f=surface.evaluate(a);H=np.empty((3*n,3*n));ledger=[dict(coordinate=None,energy=e,fmax=float(np.linalg.norm(f,axis=1).max()))]
   for k in range(3*n):
    ap=a.copy();am=a.copy();ap.positions.flat[k]+=h;am.positions.flat[k]-=h
    ep,fp=surface.evaluate(ap);em,fm=surface.evaluate(am);H[:,k]=-(fp-fm).reshape(-1)/(2*h)
    ledger.append(dict(coordinate=k,plus_energy=ep,minus_energy=em))
   t=np.tile(np.eye(3),(n,1))/np.sqrt(n);q=np.linalg.qr(t,mode='complete')[0][:,3:];sym=(H+H.T)/2;reduced=q.T@sym@q;ev=np.linalg.eigvalsh(reduced)
   row.update(status='completed',energy=e,energy_error=e-m['energy'],fmax=float(np.linalg.norm(f,axis=1).max()),hessian_skew_spectral=float(np.linalg.norm((H-H.T)/2,ord=2)),translation_residual_spectral=float(np.linalg.norm(H@t,ord=2)),eigenvalues_eV_A2=ev.tolist(),negative_eigenvalues=int(np.sum(ev<0)),ledger=ledger)
   np.savez(OUT/(arm+'.npz'),hessian=H,internal_basis=q)
  except Exception as exc:row.update(status='failed',error=repr(exc))
  row['requests']=surface.requests-before;results.append(row);(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 manifest.update(status='completed' if all(r['status']=='completed' for r in results) else 'completed_with_failures',requests=surface.requests,seconds=time.monotonic()-start)
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(requests=surface.requests,rows=[{k:r[k] for k in ('arm','status','requests')} for r in results])))
if __name__=='__main__':main()
