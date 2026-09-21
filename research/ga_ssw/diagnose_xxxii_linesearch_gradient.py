"""Actual failed stage5: frozen biased directional derivatives,13EFS maximum."""
import json,time,math,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.relax import _lbfgs_inverse_product,_accept_lbfgs_curvature
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator
from research.ga_ssw.compare_vc_arms import serial
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-central-ritz-completion';OUT=SRC/'line-search-gradient';OUT.mkdir(exist_ok=False)
d=json.loads((SRC/'result.json').read_text());record=d['result']['records'][1];cfg=json.loads((SRC/'plan.json').read_text())['config'];a=Atoms(**record['chart_reference']);top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=len(a));chart=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.);relax=record['climb'][-1]['relaxation'];q=np.array(relax['q']);g=np.array(relax['gradient']);terms=record['frozen_gaussians'];hist=[]
for l,r in zip(relax['trace'][:-1],relax['trace'][1:]):
 s=np.array(r['q'])-l['q'];y=np.array(r['gradient'])-l['gradient']
 if _accept_lbfgs_curvature(s,y):hist.append((s,y,1/float(s@y)))
hist=hist[-400:];direction=-_lbfgs_inverse_product(g,hist);direction/=np.linalg.norm(direction);random=np.random.default_rng(17).normal(size=len(q));random/=np.linalg.norm(random);directions={'lbfgs':direction,'steepest':-g/np.linalg.norm(g),'random':random}
shutil.copy2(__file__,OUT/'runner-executed.py');(OUT/'plan.json').write_text(json.dumps(dict(source=str(SRC),q=q.tolist(),directions={k:v.tolist() for k,v in directions.items()},steps=[1e-4,1e-6],expected_EFS=13,seconds=10,scope='evaluateactualfailedpoint, knownterms retained; no optimizer change or additionalsearch'),indent=2)+'\n')
class FixedG(XXXIILammpsCalculator):
 def _new_engine(self):
  from lammps import lammps
  return lammps(cmdargs=['-log',str(OUT/'engine.log'),'-screen','none'])
 def _initialize(self):
  super()._initialize()
  for command in ('pair_modify table 0','kspace_style ewald 1e-12','kspace_modify gewald 0.47570069'):self._lmp.command(command)
c=FixedG(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=a);surface=ASEStressSurface(c);start=time.monotonic()
def evaluate(x,label):
 if surface.requests>=13 or time.monotonic()-start>=10:raise RuntimeError('declared13EFS/10secondcap')
 e=chart.evaluate(x,surface.evaluate,pressure=0.);energy=e.objective;gradient=e.gradient.copy()
 for term in terms:
  n=np.array(term['direction']);z=float((x-term['center'])@n);v=term['weight']*math.exp(-.5*(z/term['width'])**2);energy+=v;gradient-=v*z/term['width']**2*n
 with (OUT/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(label=label,q=x.tolist(),atoms=serial(e.atoms),energy=e.energy,forces=e.forces.tolist(),stress=e.stress.tolist(),objective=energy,gradient=gradient.tolist()))+'\n')
 return energy,gradient,e.objective,e.gradient
try:
 center=evaluate(q,'center');rows=[]
 for name,n in directions.items():
  for h in (1e-4,1e-6):
   plus=evaluate(q+h*n,f'{name}+{h}');minus=evaluate(q-h*n,f'{name}-{h}');fd=(plus[0]-minus[0])/(2*h);analytic=float(center[1]@n)
   rows.append(dict(direction=name,h=h,analytic=analytic,FD=fd,error=fd-analytic,plus_objective_delta=plus[0]-center[0],minus_objective_delta=minus[0]-center[0],bare_FD=(plus[2]-minus[2])/(2*h),bare_analytic=float(center[3]@n)))
 out=dict(API=surface.requests,engine_calls=c.requests,initial_energy_error=center[0]-relax['energy'],initial_gradient_error=float(abs(center[1]-g).max()),rows=rows)
 (OUT/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
finally:c.close()
