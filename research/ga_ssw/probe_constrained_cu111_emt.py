"""Cu(111)+Cu adatom: one fixed-bottom SSW proposal, <=500 EF/30s CPU."""
import json,time,signal,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
from ase.build import fcc111,add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig,run_constrained_ssw
from research.ga_ssw.compare_vc_arms import serial
out=Path('research/ga_ssw/evidence/constrained-cu111-emt');out.mkdir(exist_ok=False)
a=fcc111('Cu',size=(2,2,3),a=3.6,vacuum=8.);fixed=np.flatnonzero(a.get_tags()>1);add_adsorbate(a,'Cu',height=2.,position='fcc');a.set_constraint(FixAtoms(indices=fixed));active=np.array([i for i in range(len(a)) if i not in fixed])
config=ConstrainedSSWConfig(width=.2,rotation_bias=100.,max_gaussians=2,rotation_hvp=20,relax_steps=200)
plan=dict(model='ASE EMT Cu; approximate metallic Cu slab and adatom PES, not DFT',input='ASE fcc111 Cu2x2x3 a3.6 vacuum8; add Cu fcc site height2; bottom2 layers fixed by tags>1',symbols=a.get_chemical_symbols(),positions=a.positions.tolist(),cell=a.cell.array.tolist(),pbc=a.pbc.tolist(),fixed_indices=fixed.tolist(),active_indices=active.tolist(),seed=3,steps=1,config=asdict(config),request_cap=500,seconds=30,note='development numerical settings, no search-efficiency or distinct-site claim')
(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'script.py').write_text(Path(__file__).read_text())
for name in ('constrained_reference','rc_reference','generalized_numerics','surface'):(out/(name+'.py')).write_text(Path('pamssw/standalone/'+name+'.py').read_text())
start=time.monotonic()
class Counted(ASESurface):
 def evaluate(self,a):
  if self.requests>=500 or time.monotonic()-start>30:raise RuntimeError('declared probe cap exhausted')
  e,f=super().evaluate(a)
  with (out/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(request=self.requests,energy=e,forces=f.tolist(),positions=a.positions.tolist(),cell=a.cell.array.tolist()))+'\n')
  return e,f
s=Counted(EMT());report={}
def timeout(*args):raise RuntimeError('declared probe timeout')
signal.signal(signal.SIGALRM,timeout);signal.alarm(30)
try:
 r=run_constrained_ssw(a,s,steps=1,config=config,rng=np.random.default_rng(3));report['result']=serial(r);report['fresh']=[]
 for landing in r.minima:
  raw=landing.atoms.copy();raw.set_constraint();s.calculator=EMT();e,f=s.evaluate(raw)
  report['fresh'].append(dict(energy=e,energy_error=e-landing.energy,active_fmax=float(np.linalg.norm(f[active],axis=1).max()),full_raw_fmax=float(np.linalg.norm(f,axis=1).max()),fixed_positions_exact=bool(np.array_equal(raw.positions[fixed],a.positions[fixed])),cell_exact=bool(np.array_equal(raw.cell.array,a.cell.array)),adatom_height=float(raw.positions[-1,2]-raw.positions[8:12,2].mean()),nearest_adatom_surface=float(raw.get_distances(12,list(range(12)),mic=True).min())))
except Exception as e:report.update(error=repr(e),traceback=traceback.format_exc())
finally:
 signal.alarm(0);report.update(total_requests=s.requests,seconds=time.monotonic()-start);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report.get('result',{}).get('status'),s.requests,report['seconds']);print(report.get('fresh'));print(report.get('error'))
