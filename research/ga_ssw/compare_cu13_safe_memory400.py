"""All31 frozen failed Cu13 stages: only Safe history10 ->400 changed."""
import json,time,signal,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.state import State
import pamssw.relax as relax
OUT=Path('research/ga_ssw/evidence/cu13-failed-quench-safe400')
class Budget(RuntimeError):pass
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def main():
 tasks=[]
 for p in sorted(Path('research/ga_ssw/evidence/cu13-direction-only').glob('[0-9]*-*.json')):
  for r in json.loads(p.read_text())['result']['records']:
   if r['status']=='biased_quench_failed':tasks.append((p,r))
 assert len(tasks)==31;OUT.mkdir(exist_ok=False)
 dump(OUT/'plan.json',dict(denominator=31,selection='all previously failed Cu13 biased subproblems; original starts, full Gaussian histories',change='process-local Safe memory10->400 only',per_problem_optimization_requests=201,per_problem_steps=200,fmax=.01,fresh_on_success=1,total_EF_cap=6262,wall_seconds=120,threads=1,retry=False,backend='ASE EMT Cu',reference='old Safe10 and originalELF results reused; not independent whole-search benchmark'))
 shutil.copy2(__file__,OUT/'script.py');shutil.copytree('pamssw',OUT/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 start=time.monotonic();deadline=start+120;used=0;rows=[];old=relax._SAFE_LBFGS_MEMORY;assert old==10;relax._SAFE_LBFGS_MEMORY=400
 def alarm(*_):raise Budget('120 second cap')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,120)
 try:
  for source,record in tasks:
   row=dict(source=source.name,step=record['index'],status='pending',requests=0,fresh_requests=0);rows.append(row);evals=[];accepted=[]
   g=record['climb'][-1];a=Atoms(**record['last_atoms']);a.positions=np.array(g['center'])+g['width']*np.array(g['direction']);state=State(a.numbers,a.positions)
   terms=[ProjectedGaussian(np.array(t['center']),np.array(t['direction']),t['width'],t['weight']) for t in record['climb']];surface=ASESurface(EMT())
   def evaluate(flat,template):
    nonlocal used
    if row['requests']>=201 or used>=6262 or time.monotonic()>=deadline:raise Budget('per-task/shared budget')
    row['requests']+=1;used+=1;c=Atoms(numbers=template.numbers,positions=np.asarray(flat).reshape(-1,3));e,f=surface.evaluate(c)
    for t in terms:be,bf=t.evaluate(c);e+=be;f+=bf
    evals.append(dict(positions=c.positions.tolist(),energy=e,forces=f.tolist(),max_force=float(np.linalg.norm(f,axis=1).max()),request=row['requests']));return e,-f.ravel()
   def observe(current):
    assert np.array_equal(current.positions,np.array(evals[-1]['positions']));accepted.append(evals[-1].copy())
   try:
    result=relax.Relaxer(evaluate,optimizer='safe-lbfgs-total').relax(state,fmax=.01,maxiter=200,trajectory_callback=observe);row['status']=result.telemetry.termination_reason
    if row['status']=='converged':
     if used>=6262 or time.monotonic()>=deadline:raise Budget('fresh budget')
     used+=1;row['fresh_requests']+=1;c=Atoms(numbers=a.numbers,positions=result.state.positions);e,f=ASESurface(EMT()).evaluate(c)
     for t in terms:be,bf=t.evaluate(c);e+=be;f+=bf
     row['fresh']=dict(energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),force_pass=bool(np.linalg.norm(f,axis=1).max()<=.01),energy_error=e-accepted[-1]['energy'],force_error=float(np.max(np.abs(f-np.array(accepted[-1]['forces'])))))
   except Exception as exc:row.update(status='censored' if isinstance(exc,Budget) else 'error',error=repr(exc))
   finally:
    row.update(last_accepted_force=accepted[-1]['max_force'] if accepted else None,accepted_iterates=len(accepted));dump(OUT/f'{source.stem}-step{record["index"]}.json',dict(row,evaluations=evals,accepted=accepted));dump(OUT/'summary.json',dict(runs=rows,total_requests=used,seconds=time.monotonic()-start));print(json.dumps(row),flush=True)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);relax._SAFE_LBFGS_MEMORY=old;dump(OUT/'summary.json',dict(runs=rows,denominator=31,total_requests=used,seconds=time.monotonic()-start))
if __name__=='__main__':main()
