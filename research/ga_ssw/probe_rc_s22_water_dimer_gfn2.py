"""Predeclared single water-dimer forest proposal; 400 EF/30 s total CPU cap."""
import json,time,signal,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
from ase.data.s22 import create_s22_system
from tblite.ase import TBLite
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.rc_forest_reference import RCForestSSWConfig,run_rc_forest_ssw
from research.ga_ssw.compare_vc_arms import serial
out=Path('research/ga_ssw/evidence/rc-s22-water-dimer-gfn2');out.mkdir(exist_ok=False)
a=create_s22_system('Water_dimer')
assert a.get_chemical_symbols()==['O','H','H','O','H','H']
trees=[dict(bodies=[(0,1,2)],parents=(-1,),joints=(None,)),dict(bodies=[(3,4,5)],parents=(-1,),joints=(None,))]
config=RCForestSSWConfig(rotation_length=2.,torsion_length=2.,width=.4,rotation_bias=100.,max_gaussians=2,rotation_hvp=20,relax_steps=200)
plan=dict(model='tblite GFN2-xTB accuracy .001 neutral singlet; approximate molecular PES',input='ASE S22 create_s22_system Water_dimer reference geometry; species/group order asserted OHH/OHH; see installed ase.data.s22 module provenance',positions=a.positions.tolist(),symbols=a.get_chemical_symbols(),trees=trees,anchor=0,seed=3,steps=1,config=asdict(config),request_cap=400,seconds=30,note='explicit development metric/width same as prior butane probe, not fitted defaults; no parameter tuning; numerical feasibility only')
(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'script.py').write_text(Path(__file__).read_text())
for name in ('rc_geometry','rc_reference','rc_forest','rc_forest_reference','generalized_numerics','surface'):
 (out/(name+'.py')).write_text(Path('pamssw/standalone/'+name+'.py').read_text())
start=time.monotonic()
class Counted(ASESurface):
 def evaluate(self,a):
  if self.requests>=400 or time.monotonic()-start>30:raise RuntimeError('declared probe cap exhausted')
  e,f=super().evaluate(a)
  with (out/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(request=self.requests,energy=e,forces=f.tolist(),positions=a.positions.tolist()))+'\n')
  return e,f
s=Counted(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0));report={}
def geometry(a):
 r=a.positions;dip1=(r[1]+r[2])/2-r[0];dip2=(r[4]+r[5])/2-r[3]
 contact=min(np.linalg.norm(r[i]-r[j]) for i,j in ((0,4),(0,5),(3,1),(3,2)))
 return dict(oxygen_distance=float(np.linalg.norm(r[0]-r[3])),nearest_inter_water_OH=float(contact),intramolecular_OH=[float(np.linalg.norm(r[i]-r[j])) for i,j in ((0,1),(0,2),(3,4),(3,5))],bisector_cosine=float(dip1@dip2/np.linalg.norm(dip1)/np.linalg.norm(dip2)))
def timeout(*args):raise RuntimeError('declared probe timeout')
signal.signal(signal.SIGALRM,timeout);signal.alarm(30)
try:
 r=run_rc_forest_ssw(a,s,trees=trees,anchor=0,steps=1,config=config,rng=np.random.default_rng(3));report['result']=serial(r);report['fresh']=[]
 report['geometry']=[dict(role='input',**geometry(a))]
 for k,landing in enumerate(r.minima):
  s.calculator=TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)
  e,f=s.evaluate(landing.atoms);report['fresh'].append(dict(energy=e,fmax=float(np.linalg.norm(f,axis=1).max()),energy_error=e-landing.energy));report['geometry'].append(dict(role='initial' if k==0 else 'landing',**geometry(landing.atoms)))
 for event in r.records[1:]:
  if 'last_work' in event:report['geometry'].append(dict(role='biased_last_work',**geometry(event['last_work'])))
except Exception as e:report.update(error=repr(e),traceback=traceback.format_exc())
finally:
 signal.alarm(0);report.update(total_requests=s.requests,seconds=time.monotonic()-start);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report.get('result',{}).get('status'),s.requests,report['seconds']);print(report.get('geometry'));print(report.get('error'))
