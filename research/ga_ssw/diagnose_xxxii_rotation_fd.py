"""Fixed failed RC-VC direction: forward vs centered finite differences, no search."""
import json,time,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator
from research.ga_ssw.compare_vc_arms import serial
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-globalbudget-memory400';OUT=SRC/'rotation-fd-check';OUT.mkdir(exist_ok=False)
d=json.loads((SRC/'result.json').read_text());record=d['result']['records'][1];cfg=json.loads((SRC/'plan.json').read_text())['config'];a=Atoms(**record['chart_reference']);top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=len(a));chart=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
q=np.array(record['climb'][-2]['scaled_coordinates']);n=np.array(record['climb'][-1]['mode']['direction']);anchor=np.random.default_rng(3).normal(size=chart.dimension);anchor/=np.linalg.norm(anchor);beta=cfg['rotation_bias']
shutil.copy2(__file__,OUT/'runner-executed.py');(OUT/'plan.json').write_text(json.dumps(dict(source=str(SRC),q=q.tolist(),direction=n.tolist(),anchor=anchor.tolist(),steps=[1e-4,5e-5,2.5e-5],expected_EFS=13,max_EFS=13,seconds=10,scope='fixed original failed direction only; no replacement solver or tolerance change; known erfc finite precision retained'),indent=2)+'\n')
class FixedG(XXXIILammpsCalculator):
 def _new_engine(self):
  from lammps import lammps
  return lammps(cmdargs=['-log',str(OUT/'engine.log'),'-screen','none'])
 def _initialize(self):
  super()._initialize()
  for command in ('pair_modify table 0','kspace_style ewald 1e-12','kspace_modify gewald 0.47570069'):self._lmp.command(command)
c=FixedG(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=a);s=ASEStressSurface(c);start=time.monotonic()
def grad(x,label):
 if s.requests>=13 or time.monotonic()-start>10:raise RuntimeError('declared diagnostic budget')
 e=chart.evaluate(x,s.evaluate,pressure=cfg['pressure'])
 with (OUT/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(label=label,q=x.tolist(),atoms=serial(e.atoms),energy=e.energy,forces=e.forces.tolist(),stress=e.stress.tolist(),gradient=e.gradient.tolist()))+'\n')
 return e.gradient
try:
 g0=grad(q,'center');rows=[];t=None
 for h in (1e-4,5e-5,2.5e-5):
  gp=grad(q+h*n,f'n+{h}');gm=grad(q-h*n,f'n-{h}');hf=(gp-g0)/h-beta*(anchor@n)*anchor;hc=(gp-gm)/(2*h)-beta*(anchor@n)*anchor
  if t is None:
   t=hc-n*(n@hc);t/=np.linalg.norm(t)
  tp=grad(q+h*t,f't+{h}');tm=grad(q-h*t,f't-{h}');tf=(tp-g0)/h-beta*(anchor@t)*anchor;tc=(tp-tm)/(2*h)-beta*(anchor@t)*anchor
  rows.append(dict(h=h,forward_residual=float(np.linalg.norm(hf-n*(n@hf))),central_residual=float(np.linalg.norm(hc-n*(n@hc))),forward_curvature=float(n@hf),central_curvature=float(n@hc),hvp_difference=float(np.linalg.norm(hf-hc)),forward_projected_asymmetry=float(abs(n@tf-t@hf)),central_projected_asymmetry=float(abs(n@tc-t@hc))))
 out=dict(rows=rows,reported_mode={k:v for k,v in record['climb'][-1]['mode'].items() if k!='direction'},API=s.requests,engine_calls=c.requests,seconds=time.monotonic()-start)
 (OUT/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
finally:c.close()
