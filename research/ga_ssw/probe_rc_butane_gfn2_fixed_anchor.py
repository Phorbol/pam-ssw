"""One bounded real-molecule RC pipeline probe: 400 EF/30seconds, no efficacy claim."""
import json,time,signal,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
from ase.build import molecule
from tblite.ase import TBLite
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.rc_reference import RCSSWConfig,run_rc_ssw
from research.ga_ssw.compare_vc_arms import serial
out=Path('research/ga_ssw/evidence/rc-butane-gfn2-fixed-anchor');out.mkdir(exist_ok=False)
config=RCSSWConfig(torsion_length=2.,width=.4,rotation_bias=100.,max_gaussians=2,rotation_hvp=20,relax_steps=200)
bodies=[(0,1,4,6,7),(0,1,2,10,11),(1,2,3,5,8,9,12,13)];parents=(-1,0,1);joints=(None,(0,1),(1,2));a=molecule('trans-butane')
plan=dict(model='tblite GFN2-xTB, accuracy .001; neutral singlet molecular model',initial='ASE trans-butane',seed=3,steps=1,config=asdict(config),request_cap=400,seconds=30,bodies=bodies,parents=parents,joints=joints,note='metric2 A/rad and width .4A are explicit development inputs, not fitted optimum; 2 Gaussian cap numerical feasibility only')
(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'script.py').write_text(Path(__file__).read_text());(out/'rc_reference.py').write_text(Path('pamssw/standalone/rc_reference.py').read_text());(out/'rc_geometry.py').write_text(Path('pamssw/standalone/rc_geometry.py').read_text())
start=time.monotonic()
class Counted(ASESurface):
 def evaluate(self,a):
  if self.requests>=400 or time.monotonic()-start>30:raise RuntimeError('declared probe cap exhausted')
  e,f=super().evaluate(a)
  with (out/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(request=self.requests,energy=e,forces=f.tolist(),positions=a.positions.tolist()))+'\n')
  return e,f
s=Counted(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0));report={}
def timeout(*a):raise RuntimeError('declared probe timeout')
signal.signal(signal.SIGALRM,timeout);signal.alarm(30)
try:
 r=run_rc_ssw(a,s,bodies=bodies,parents=parents,joints=joints,steps=1,config=config,rng=np.random.default_rng(3));report['result']=serial(r);report['fresh']=[]
 for landing in r.minima:
  # Independent calculator instance, same total budget and explicit ledger.
  s.calculator=TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)
  e,f=s.evaluate(landing.atoms);report['fresh'].append(dict(energy=e,fmax=float(np.linalg.norm(f,axis=1).max()),energy_error=e-landing.energy))
except Exception as e:report.update(error=repr(e),traceback=traceback.format_exc())
finally:
 signal.alarm(0);report.update(total_requests=s.requests,seconds=time.monotonic()-start);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report.get('result',{}).get('status'),s.requests,report['seconds'])
