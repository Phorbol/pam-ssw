"""Bounded continuation of the saved phase87 PQC completed-Gaussian boundary."""
import hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
from ase import Atoms
BASE=Path('research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step').resolve()
OLD=BASE/'pqc'; OUT=BASE/'pqc-continuation'
OUT.mkdir(exist_ok=False)
shutil.copytree(OLD/'source', OUT/'source')
shutil.copy2(__file__, OUT/'runner-executed.py')
sys.path.insert(0,str(OUT/'source'))
from pamssw.standalone.atomic_climb import AtomicClimbCheckpoint, resume_atomic_climb
import pamssw.standalone.atomic_climb as unused
import importlib
module=importlib.import_module('pamssw.standalone.atomic_climb')
from pamssw.standalone.paper_reference import SSWConfig, sample_initial_direction
from pamssw.standalone.block_ssw import FixedCellSurface
from pamssw.standalone.cell_relax import cell_quench
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.compare_vc_arms import serial, mc_accept, BudgetExhausted
r=json.loads((OLD/'result.json').read_text()); cp=r['records'][1]['atomic']['checkpoint']
config=SSWConfig(**cp['config']); initial=r['landings'][0]; b=r['block_config']
rng=np.random.default_rng(r['seed'])
anchor=sample_initial_direction(Atoms(**initial['atoms']),rng,mode=config.direction_sampling)
assert np.array_equal(anchor,np.asarray(cp['initial_anchor']))
checkpoint=AtomicClimbCheckpoint(cp['next_index'],Atoms(**cp['atoms']),anchor,
 tuple(cp['climb']),cp['reference_energy'],config,cp['previous_requests'],cp['pending'],cp['terminal_status'])
model='/home/gengjianrui/.cache/mace/mace-omat-0-small.model'
plan=dict(original_result=str(OLD/'result.json'),original_sha256=hashlib.sha256((OLD/'result.json').read_bytes()).hexdigest(),
 original_requests=r['requests'], original_status=r['status'],boundary=cp['next_index'],
 additional_cap=1000,additional_seconds=600,config=cp['config'],block_config=b,
 model=model,model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
 domain='fixed-cell escape plus posterior cell quench; NOT joint escape',
 policy='redo pending Gaussian at charged cost; no history or initial direction resampling; preserve original censoring result',
 source=str(OUT/'source'),anchor_max_error=0.0)
(OUT/'plan.json').write_text(json.dumps(plan,indent=2))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from mace.calculators import MACECalculator
calc=MACECalculator(model_paths=model,device='cpu',default_dtype='float64')
start=time.monotonic()
class BoundedSurface(ASEStressSurface):
 exhausted=False
 def evaluate(self,a):
  if self.requests>=1000 or time.monotonic()-start>=600:
   self.exhausted=True;raise BudgetExhausted('additional continuation cap')
  before=self.requests
  try:
   value=super().evaluate(a)
  except Exception as exc:
   with (OUT/'ledger.jsonl').open('a') as f:f.write(json.dumps(dict(before=before,after=self.requests,error=repr(exc)))+'\n')
   raise
  with (OUT/'ledger.jsonl').open('a') as f:f.write(json.dumps(dict(before=before,after=self.requests,energy=value[0]))+'\n')
  return value
surface=BoundedSurface(calc);checks=[];original_quench=module.quench
# Validate the first regenerated stage before it is optimized.
def checked_quench(atoms,surface,**kwargs):
 if not checks:
  pending=cp['pending'];term=kwargs['terms'][-1]
  check=dict(position_error=float(np.max(abs(atoms.positions-np.asarray(pending['displaced']['positions'])))),
   direction_error=float(np.max(abs(term.direction-np.asarray(pending['direction'])))),
   weight_error=float(abs(term.weight-pending['weight'])))
  checks.append(check);(OUT/'pending-comparison.json').write_text(json.dumps(check,indent=2))
  if check['position_error']>1e-7 or check['direction_error']>1e-7 or check['weight_error']>1e-7:
   raise RuntimeError('regenerated pending stage differs; stop before optimizing')
 return original_quench(atoms,surface,**kwargs)
module.quench=checked_quench
result=dict(status='running',original_requests=r['requests'],landing=None,accepted=False)
try:
 climb=resume_atomic_climb(checkpoint,FixedCellSurface(surface))
 result['climb']=climb;result['status']=climb.status
 if climb.status in ('gaussian_limit','lower_true_energy'):
  landing=cell_quench(climb.atoms,surface,strain_length=b['quench_length'],pressure=b['pressure'],
    fmax=config.fmax,stress_tol=b['stress_tol'],max_step=b['max_step'],maxiter=config.relax_steps)
  result['landing']=landing;result['status']='valid_landing' if landing.converged else 'true_quench_failed'
  if landing.converged:
   delta=landing.evaluation.objective-initial['objective']
   accepted,draw=mc_accept(delta,config.temperature_K,rng)
   result.update(delta=delta,accepted=bool(accepted),mc_draw=draw)
   calc.reset();e,f,s=surface.evaluate(landing.evaluation.atoms)
   result['fresh']=dict(energy=e,fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(abs(s+b['pressure']*np.eye(3)).max()),forces=f,stress=s)
except Exception as exc:
 result.update(status='censored' if surface.exhausted else 'failed',error=repr(exc))
result.update(additional_requests=surface.requests,total_requests=r['requests']+surface.requests,wall_seconds=time.monotonic()-start,pending_comparisons=checks)
(OUT/'result.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
print({k:v for k,v in result.items() if k not in ('climb','landing','fresh')},flush=True)
