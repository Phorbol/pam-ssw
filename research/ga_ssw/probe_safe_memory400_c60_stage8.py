"""Single-factor selected-stage ablation: Safe memory10 to native-derived400."""
import json,time,signal,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.surface import ASESurface,quench
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.standalone.softening import FrozenBondSoftening
import pamssw.relax as relax
from research.ga_ssw.compare_vc_arms import serial
P=Path('research/ga_ssw/evidence/hard-c60-gfn2-one-step-2000/paper-seed3')
OUT=Path('research/ga_ssw/evidence/hard-c60-safe-memory400-stage8')
class Budget(RuntimeError):pass
def dump(p,d):p.write_text(json.dumps(serial(d),indent=2,allow_nan=False))
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 frozen=json.loads((P/'offline-stage8/frozen-objective.json').read_text());a=Atoms(**frozen['start']);soft=FrozenBondSoftening(**frozen['soft']);terms=[soft]+[ProjectedGaussian(np.array(c['center']),np.array(c['direction']),c['width'],c['weight']) for c in frozen['gaussians']]
 assert relax._SAFE_LBFGS_MEMORY==10
 dump(OUT/'plan.json',dict(hypothesis='Is memory10 to native-derived400 alone sufficient to change convergence of this selected failure? No optimal-history claim.',change='process-local pamssw.relax._SAFE_LBFGS_MEMORY=400 only; no public module changes',fmax=.01,accepted_step_cap=400,optimization_EF_cap=423,total_EF_cap=424,wall_seconds=600,threads=1,backend='tblite0.7 GFN2-xTB accuracy .001',source='same original frozen stage8 start',reference='archived Safe10 423EF400steps; original native314EF309steps; neither rerun on PES',unchanged_constants={k:v for k,v in vars(relax).items() if k.startswith('_SAFE_LBFGS') and isinstance(v,(int,float,str))}))
 shutil.copy2(__file__,OUT/'script.py');shutil.copy2(P/'offline-stage8/frozen-objective.json',OUT/'frozen-objective.json');shutil.copy2(P/'offline-stage8/summary.json',OUT/'memory10-exact-offline-replay.json')
 shutil.copytree('pamssw',OUT/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 from tblite.ase import TBLite
 started=time.monotonic();deadline=started+600;used=0;rows=[];accepted=[];error=None;status='running';q=None;fresh=None
 def alarm(*_):raise Budget('600 second wall cap')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,600)
 log=(OUT/'evaluations.jsonl').open('w')
 class Counted(ASESurface):
  def __init__(self,fresh=False):super().__init__(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0));self.fresh=fresh
  def evaluate(self,atoms):
   nonlocal used
   if used >= (424 if self.fresh else 423) or time.monotonic()>=deadline:raise Budget('request/wall cap')
   used+=1
   try:
    e,f=super().evaluate(atoms);te=e;tf=f.copy()
    for term in terms:be,bf=term.evaluate(atoms);te+=be;tf+=bf
    row=dict(request=used,fresh=self.fresh,positions=atoms.positions.tolist(),raw_energy=e,raw_forces=f.tolist(),energy=te,forces=tf.tolist(),max_force=float(np.linalg.norm(tf,axis=1).max()));rows.append(row);log.write(json.dumps(row)+'\n');log.flush();return e,f
   except Exception as exc:log.write(json.dumps(dict(request=used,error=repr(exc)))+'\n');log.flush();raise
 original_memory=relax._SAFE_LBFGS_MEMORY;original_accept=relax._accept_lbfgs_curvature
 def capture(s,y):
  ok=original_accept(s,y);accepted.append(dict(rows[-1],s_dot_y=float(s@y),retained=bool(ok)));return ok
 relax._SAFE_LBFGS_MEMORY=400;relax._accept_lbfgs_curvature=capture
 try:
  q=quench(a,Counted(),fmax=.01,steps=400,terms=terms,optimizer='safe-lbfgs-total');status='converged' if q.converged else 'step_limit';dump(OUT/'quench.json',q)
 except Exception as exc:error=repr(exc);status='budget' if used>=423 or time.monotonic()>=deadline or isinstance(exc,Budget) else 'error'
 finally:
  relax._SAFE_LBFGS_MEMORY=original_memory;relax._accept_lbfgs_curvature=original_accept
  dump(OUT/'pre-fresh.json',dict(status=status,error=error,physical_requests=used,accepted=accepted))
 try:
  endpoint=accepted[-1] if accepted else rows[0];a.positions=np.array(endpoint['positions']);e,f=Counted(True).evaluate(a);fresh=rows[-1];fresh['energy_error']=fresh['energy']-endpoint['energy'];fresh['force_error']=float(np.max(np.abs(np.array(fresh['forces'])-np.array(endpoint['forces']))))
 except Exception as exc:fresh=dict(error=repr(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);log.close();summary=dict(status=status,error=error,physical_requests=used,accepted_steps=len(accepted),negative_pairs=sum(x['s_dot_y']<0 for x in accepted),rejected_curvature=sum(not x['retained'] for x in accepted),fresh=fresh,seconds=time.monotonic()-started);dump(OUT/'result.json',dict(summary,accepted=accepted));print(json.dumps({**summary,'fresh':{k:v for k,v in fresh.items() if k not in ['positions','forces','raw_forces']}}),flush=True)
if __name__=='__main__':main()
