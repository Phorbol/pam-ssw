"""Resume a censored material atomic stage; retain original-run accounting."""
import argparse,dataclasses,json,shutil,time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from pamssw.standalone.atomic_climb import AtomicClimbCheckpoint,resume_atomic_climb
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.block_ssw import FixedCellSurface
from pamssw.standalone.cell_relax import cell_quench
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.compare_vc_arms import serial


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
 p.add_argument('--seconds',type=float,default=600);p.add_argument('--cap',type=int,default=1500);args=p.parse_args()
 src=Path(args.source);data=json.loads((src/'result.json').read_text());plan=json.loads((src/'plan.json').read_text());outer=data['result'];stage=outer['records'][-1]['atomic'];cfg=SSWConfig(**plan['config']['atomic'])
 if Path(args.model).resolve()!=Path(plan['arguments']['model']).resolve():raise ValueError('replay requires the original model path')
 work=Atoms(**stage['atoms']);complete=[e for e in stage['climb'] if 'true_energy' in e]
 if len(complete)!=sum('center' in e for e in stage['climb']):raise ValueError('old artifact has ambiguous completed boundary')
 before=outer['initial']['evaluation']
 for ev in outer['records'][1:-1]:
  if ev['accepted']:before=ev['landing']['evaluation']
 reference=before['objective']-plan['config']['pressure']*work.get_volume()
 checkpoint=AtomicClimbCheckpoint(len(complete),work,np.array(stage['initial_direction']),tuple(complete),reference,cfg,stage['requests'],pending=dict(stage='old_artifact_pending_not_recorded'),terminal_status=None)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 (out/'plan.json').write_text(json.dumps(dict(arguments=vars(args),original_search_requests=data['search_requests'],original_fresh_requests=data['fresh_requests'],resume_gaussian_index=len(complete),source_config=plan['config'],scope='atomic-stage replay plus optional final quench, no retrospective MC or original-status change'),indent=2))
 shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(__file__,out/'script.py')
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from mace.calculators import MACECalculator
 calc=MACECalculator(model_paths=args.model,device='cpu',default_dtype='float64')
 start=time.monotonic()
 class Surface(ASEStressSurface):
  def evaluate(self,a):
   if self.requests>=args.cap or time.monotonic()-start>=args.seconds:raise RuntimeError('declared replay request/wall budget exhausted')
   v=super().evaluate(a)
   if self.requests%25==0:(out/'progress.json').write_text(json.dumps(dict(requests=self.requests,seconds=time.monotonic()-start,energy=v[0],fmax=float(np.linalg.norm(v[1],axis=1).max()))))
   return v
 surface=Surface(calc)
 result=resume_atomic_climb(checkpoint,FixedCellSurface(surface))
 payload=dict(atomic=serial(result),original_experiment_status='retained_censored',fresh_requests=0)
 if result.status in ('lower_true_energy','gaussian_limit'):
  block=plan['config']
  landing=cell_quench(result.atoms,surface,strain_length=block['quench_length'],pressure=block['pressure'],fmax=cfg.fmax,stress_tol=block['stress_tol'],max_step=block['max_step'],maxiter=cfg.relax_steps)
  payload['landing']=serial(landing)
  if landing.converged:
   write(out/'landing.extxyz',landing.evaluation.atoms)
   payload.update(new_search_requests=surface.requests,replay_seconds=time.monotonic()-start,accumulated_search_requests=data['search_requests']+surface.requests)
   (out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
   # One separately accounted certification request; preserve search if it fails.
   fresh=ASEStressSurface(calc)
   try:
    if time.monotonic()-start>=args.seconds:raise RuntimeError('replay wall budget exhausted before fresh certificate')
    calc.reset();e,f,s=fresh.evaluate(landing.evaluation.atoms)
    payload['fresh_certificate']=dict(energy_error=e-landing.evaluation.energy,fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(np.abs(s+block['pressure']*np.eye(3)).max()))
   except Exception as error:
    payload['fresh_error']=repr(error)
   finally:
    payload['fresh_requests']=fresh.requests
 payload.update(new_search_requests=surface.requests,replay_seconds=time.monotonic()-start,
                accumulated_search_requests=data['search_requests']+surface.requests)
 (out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
 print(json.dumps({k:v for k,v in payload.items() if k not in ('atomic','landing')}))
if __name__=='__main__':main()
