"""Matched fixed first RC-VC bias: existing memory10 result vs new memory400.

No new search policy: same displaced q, chart, bias, tolerances and300-stepcap.
"""
import json,math,time,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.generalized_numerics import safe_lbfgs
from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator
from research.ga_ssw.compare_vc_arms import serial
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-one-step';OUT=SRC/'frozen-memory400';OUT.mkdir(exist_ok=False)
d=json.loads((SRC/'result.json').read_text());ev=d['result']['records'][1];cfg=json.loads((SRC/'plan.json').read_text())['config']
a=Atoms(**ev['chart_reference']);top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=len(a));chart=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=cfg['rotation_length'],torsion_length=cfg['torsion_length'],strain_length=cfg['strain_length'])
old=ev['climb'][0]['relaxation'];q0=np.array(old['trace'][0]['q']);terms=ev['frozen_gaussians'];assert len(terms)==1
for name in ('generalized_numerics','rc_optimization_domain','rc_vc_geometry','rc_forest','rc_geometry','vc_geometry'):
 p=ROOT/f'pamssw/standalone/{name}.py';assert p.read_bytes()==(SRC/f'source/pamssw/standalone/{name}.py').read_bytes();shutil.copy2(p,OUT/p.name)
shutil.copy2(__file__,OUT/'runner-executed.py')
(OUT/'plan.json').write_text(json.dumps(dict(source=str(SRC),changed={'lbfgs_memory':[None,400]},same=['q0','chart','firstGaussian','300steps','.005norm','.2maxstep','calculator'],q0=q0.tolist(),terms=terms,max_EFS=500,max_seconds=30,claim='frozen biased optimization only; not whole SS W step or general efficiency; previous memory10 costs retained',reason='same positive secants but memory10 stalls at.331 gradient; test retained curvature information without newheuristic'),indent=2)+'\n')
class FixedG(XXXIILammpsCalculator):
 def _new_engine(self):
  from lammps import lammps
  return lammps(cmdargs=['-log',str(OUT/'engine.log'),'-screen','none'])
 def _initialize(self):
  super()._initialize()
  for command in ('pair_modify table 0','kspace_style ewald 1e-12','kspace_modify gewald 0.47570069'):self._lmp.command(command)
c=FixedG(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=a);s=ASEStressSurface(c);start=time.monotonic();initial_check={}
def evaluate(q):
 if s.requests>=500 or time.monotonic()-start>=30:raise RuntimeError('declared500EFS/30sec cap')
 x=chart.evaluate(q,s.evaluate,pressure=cfg['pressure']);energy=x.objective;g=x.gradient.copy()
 for term in terms:
  n=np.array(term['direction']);z=float((q-term['center'])@n);v=term['weight']*math.exp(-.5*(z/term['width'])**2);energy+=v;g-=v*z/term['width']**2*n
 with (OUT/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(q=q.tolist(),atoms=serial(x.atoms),energy=x.energy,forces=x.forces.tolist(),stress=x.stress.tolist(),objective=energy,gradient=g.tolist()))+'\n')
 if not initial_check:
  initial_check.update(energy_error=energy-old['trace'][0]['energy'],gradient_max_error=float(abs(g-old['trace'][0]['gradient']).max()))
  assert abs(initial_check['energy_error'])<1e-10 and initial_check['gradient_max_error']<1e-9
 return energy,g
try:
 r=safe_lbfgs(q0,evaluate,gradient_norm=np.linalg.norm,step_norm=np.linalg.norm,gtol=cfg['gradient_tol'],max_step=cfg['max_step'],maxiter=cfg['relax_steps'],lbfgs_memory=400)
 report=dict(initial_check=initial_check,old_memory10={k:old[k] for k in ('status','steps','requests','accepted_secants','rejected_secants','rejected_trials')},new_memory400=serial(r),api_requests=s.requests,engine_calls=c.requests,seconds=time.monotonic()-start)
 (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(r.status,r.steps,s.requests,float(np.linalg.norm(r.gradient)),initial_check)
finally:c.close()
