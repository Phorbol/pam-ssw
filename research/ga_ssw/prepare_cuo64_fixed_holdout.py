"""Prepared bounded CuO64 fixed-cell holdout; requires explicit --execute."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; EVIDENCE=ROOT/'research/ga_ssw/evidence/cuo64-vc-input-qualification/qualification'; MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model'); SEEDS=(11,29); CAP=1200; WALL=300
class BudgetStop(RuntimeError): pass
def cfg(S, C): return C(width=.1,rotation_bias=None if S else 100.,pre_rotation_hvp=5 if S else None,max_gaussians=25,temperature_K=150.,fmax=.01,bias_fmax=.1,relax_steps=400,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,direction_sampling='global',rotation_solver='ritz',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
def main(execute=False):
    out=ROOT/'research/ga_ssw/evidence/cuo64-fixed-ritz-staged-holdout-20260912'; out.mkdir(parents=False,exist_ok=False)
    shutil.copytree(ROOT/'pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc')); (out/'input-evidence').mkdir(); [shutil.copy2(EVIDENCE/n,out/'input-evidence'/n) for n in ('quench-endpoint.json','result.json')]; shutil.copy2(__file__,out/'runner.py')
    common={'width':.1,'max_gaussians':25,'temperature_K':150.,'fmax':.01,'bias_fmax':.1,'relax_steps':400,'fd_step':1e-4,'rotation_hvp':100,'rotation_tol':.02,'direction_sampling':'global','rotation_solver':'ritz','cluster_frame':'translation_only','quench_optimizer':'safe-lbfgs-total'}
    manifest={'status':'prepared_not_executed' if not execute else 'running','source':str(EVIDENCE/'quench-endpoint.json'),'source_url':'https://aflow.org/p/AB_mC8_15_a_e-001/aflow.cif','input_history':'AFLOW tenorite 8 atoms repeated 2x2x2; archived joint-cell input quench and Hessian qualification on V100; this campaign freezes that endpoint cell and runs only on CPU','model':str(MODEL),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'seeds':list(SEEDS),'arms':['fixed_ritz_a100','staged_ritz_pre5_aactual'],'steps':100,'request_cap_inclusive':CAP,'wall_cap_seconds_per_arm':WALL,'config_fixed':dict(common,rotation_bias=100.,pre_rotation_hvp=None),'config_staged':dict(common,rotation_bias=None,pre_rotation_hvp=5),'certificate':'fresh single E/F evaluation per completed landing; geometry unchanged','environment':{k:os.environ.get(k) for k in ('PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')},'claim_boundary':'fixed-cell MACE-OMAT comparison only'}
    if not execute: (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); return
    sys.path.insert(0,str(out/'source'))
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from ase import Atoms
    from ase.io import write
    from mace.calculators import MACECalculator
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import ASESurface
    import pamssw
    manifest['import_path']=pamssw.__file__
    assert str(out/'source') in str(pamssw.__file__)
    from research.ga_ssw.compare_vc_arms import serial
    manifest['configs']={'fixed':serial(cfg(False,SSWConfig)),'staged':serial(cfg(True,SSWConfig))}; (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for system in ('cuo64',):
      atoms=Atoms(**json.loads((out/'input-evidence'/'quench-endpoint.json').read_text())); write(out/f'{system}.extxyz',atoms)
      for seed in SEEDS:
       for arm,S in (('fixed_ritz_a100',False),('staged_ritz_pre5_aactual',True)):
        folder=out/f'{system}-seed{seed}-{arm}'; folder.mkdir(); start=time.monotonic(); log=[]
        class Counted(ASESurface):
          boundary = None
          def evaluate(self,a):
            if self.requests>=CAP:
              self.boundary='request_cap'; raise BudgetStop(self.boundary)
            if time.monotonic()-start>=WALL:
              self.boundary='wall_cap'; raise BudgetStop(self.boundary)
            try:
              e,f=super().evaluate(a); item={'request':self.requests,'energy':e,'fmax':float(np.linalg.norm(f,axis=1).max()),'positions':a.positions.tolist()}
            except Exception as exc:
              item={'request':self.requests,'error':repr(exc),'positions':a.positions.tolist()}
              log.append(item)
              with (folder/'evaluations.jsonl').open('a') as stream: stream.write(json.dumps(item)+'\n')
              raise
            log.append(item)
            with (folder/'evaluations.jsonl').open('a') as stream: stream.write(json.dumps(item)+'\n')
            return e,f
        surface=Counted(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)); row={'system':system,'seed':seed,'arm':arm}
        try:
          r=run_ssw(atoms,surface,steps=100,config=cfg(S,SSWConfig),rng=np.random.default_rng(seed)); (folder/'result.json').write_text(json.dumps(serial(r),indent=2)+'\n'); cert_calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False); fresh=ASESurface(cert_calc); checks=[]
          for i,q in enumerate(r.minima):
            check={'index':i}
            original=q.atoms.positions.copy(); cert_calc.reset()
            try:
              e,f=fresh.evaluate(q.atoms); check.update(energy=e,energy_error=e-q.energy,fmax=float(np.linalg.norm(f,axis=1).max()),requests=fresh.requests,cell_unchanged=bool(np.array_equal(atoms.cell.array,q.atoms.cell.array)),geometry_unchanged=bool(np.array_equal(original,q.atoms.positions) and np.array_equal(atoms.cell.array,q.atoms.cell.array) and np.array_equal(atoms.pbc,q.atoms.pbc)))
            except Exception as exc: check.update(status='failed',error=repr(exc),requests=fresh.requests)
            finally:
              checks.append(check); (folder/'fresh-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
          write(folder/'minima.extxyz',[q.atoms for q in r.minima]); row.update(status=r.status,search_requests=surface.requests,ledger_count=len(log),records=len(r.records),fresh_checks=checks,fresh_requests=fresh.requests)
        except Exception as e: row.update(status=type(e).__name__,error=repr(e),search_requests=surface.requests)
        row['ledger_count']=len(log)
        row['boundary']=surface.boundary
        row['seconds']=time.monotonic()-start
        row['accounted']=row['ledger_count']==surface.requests
        (folder/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
    manifest['status']='completed'; (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); main(p.parse_args().execute)
