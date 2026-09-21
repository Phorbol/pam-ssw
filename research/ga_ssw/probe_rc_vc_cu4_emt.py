"""Bounded actual-ASE RC-VC wiring, artificial rigid Cu grouping not molecules."""
import json,time,signal,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.rc_vc_reference import RCVCSSWConfig,run_rc_vc_ssw
from research.ga_ssw.compare_vc_arms import serial
out=Path('research/ga_ssw/evidence/rc-vc-cu4-emt');out.mkdir(exist_ok=False)
a=bulk('Cu','fcc',a=3.6,cubic=True)
trees=[dict(bodies=[tuple(range(4))],parents=(-1,),joints=(None,))]
config=RCVCSSWConfig(rotation_length=2.,torsion_length=2.,strain_length=4.,width=.2,rotation_bias=100.,max_gaussians=1,rotation_hvp=20,relax_steps=200)
plan=dict(model='ASE EMT Cu; physically metal potential but artificial rigid four-Cu group is numerical wiring only, no molecular material claim',input='ASE bulk Cu fcc conventional4 a3.6A',positions=a.positions.tolist(),cell=a.cell.array.tolist(),trees=trees,anchor=0,seed=3,steps=1,config=asdict(config),request_cap=400,seconds=30,note='Explicit development metric and one-Gaussian budget, no tuning or efficacy inference')
(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'script.py').write_text(Path(__file__).read_text())
for name in ('rc_geometry','rc_forest','rc_vc_geometry','rc_vc_reference','generalized_numerics','cell_relax','vc_geometry'):
 (out/(name+'.py')).write_text(Path('pamssw/standalone/'+name+'.py').read_text())
start=time.monotonic()
class Counted(ASEStressSurface):
 def evaluate(self,a):
  if self.requests>=400 or time.monotonic()-start>30:raise RuntimeError('declared probe cap exhausted')
  e,f,s=super().evaluate(a)
  with (out/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(request=self.requests,energy=e,forces=f.tolist(),stress=s.tolist(),positions=a.positions.tolist(),cell=a.cell.array.tolist()))+'\n')
  return e,f,s
s=Counted(EMT());report={}
def timeout(*args):raise RuntimeError('declared probe timeout')
signal.signal(signal.SIGALRM,timeout);signal.alarm(30)
try:
 r=run_rc_vc_ssw(a,s,trees=trees,anchor=0,steps=1,config=config,rng=np.random.default_rng(3));report['result']=serial(r);report['fresh']=[]
 for landing in r.minima:
  s.calculator=EMT();e,f,stress=s.evaluate(landing.atoms)
  report['fresh'].append(dict(energy=e,fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(np.abs(stress+config.pressure*np.eye(3)).max()),energy_error=e-landing.energy,volume=landing.atoms.get_volume()))
except Exception as e:report.update(error=repr(e),traceback=traceback.format_exc())
finally:
 signal.alarm(0);report.update(total_requests=s.requests,seconds=time.monotonic()-start);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report.get('result',{}).get('status'),s.requests,report['seconds']);print(report.get('fresh'));print(report.get('error'))
