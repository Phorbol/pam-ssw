"""One selected frozen LS failure: original ELF kernel versus archived Safe-total."""
import json,time,signal,shutil,hashlib
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.standalone.softening import FrozenBondSoftening
from research.ga_ssw.probe_native_lbfgs_emt import Oracle,SETTINGS,ELF,load_elf,DATA
from research.ga_ssw.compare_vc_arms import serial
P=Path('research/ga_ssw/evidence/hard-c60-gfn2-one-step-2000/paper-seed3')
OUT=Path('research/ga_ssw/evidence/hard-c60-native-stage8')
class Budget(RuntimeError):pass
class LargeOracle(Oracle):
 def alloc(self,data):
  if self.cursor+len(data)>DATA+0x100000 and not getattr(self,'extended',False):
   self.uc.mem_map(DATA+0x100000,0x100000);self.extended=True
  return super().alloc(data)
 def hook(self,u,pc,size,data):
  if pc==0x6e8dad and self.integer(0x791b938)==1:
   mismatch=float(np.max(np.abs(self.x()-np.array(self.current['positions']))))
   if mismatch>1e-12:raise ValueError('accepted coordinate mismatch')
   self.accepted.append(dict(self.current,accepted_coordinate_error=mismatch))
   if self.current['max_force']<=.01:self.stop_reason='converged'
   elif len(self.accepted)>=400:self.stop_reason='step_limit'
   if self.stop_reason:u.emu_stop()
   return
  return super().hook(u,pc,size,data)
def dump(p,d):p.write_text(json.dumps(serial(d),indent=2,allow_nan=False))
def main():
 OUT.mkdir(parents=True,exist_ok=False);blob,segments=load_elf(ELF)
 sha=hashlib.sha256(blob).hexdigest();assert sha=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
 frozen=json.loads((P/'offline-stage8/frozen-objective.json').read_text());a=Atoms(**frozen['start']);soft=FrozenBondSoftening(**frozen['soft']);terms=[soft]+[ProjectedGaussian(np.array(c['center']),np.array(c['direction']),c['width'],c['weight']) for c in frozen['gaussians']]
 dump(OUT/'plan.json',dict(settings=dict(SETTINGS,request_cap=423,step_cap=400),total_physical_EF_cap=424,fresh_reserve=1,wall_cap_seconds=600,threads=1,elf=ELF,sha256=sha,backend='tblite0.7 GFN2-xTB accuracy .001',selection='selected failed eighth stage only; no full walker rerun',oracle_changes='extra DATA map for N180 workspace; accepted step cap400; accepted-coordinate assertion; numerical settings unchanged',reference='prior31Cu kernel replay settings GTOL900/history400; gradient scale1, no BFGSDRIVER .05 scaling; initialized ELF GTOL not claim all runtime configurations',safe_reference_requests=423,safe_reference_steps=400,safe_reference_fmax=.011354581020178715))
 shutil.copy2(P/'offline-stage8/frozen-objective.json',OUT/'frozen-objective.json');shutil.copy2(__file__,OUT/'script.py')
 shutil.copytree('pamssw',OUT/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copy2('research/ga_ssw/probe_native_lbfgs_emt.py',OUT/'oracle-source.py');shutil.copy2('research/ga_ssw/probe_native_weight_emulated.py',OUT/'elf-loader-source.py')
 from tblite.ase import TBLite
 def surface():return ASESurface(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0))
 s=surface();oracle=LargeOracle(segments,a.positions);used=0;rows=[];status='request_limit';error=None;fresh=None;flag=None
 start=time.monotonic();deadline=start+600
 def alarm(*_):raise Budget('600 second cap')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,600)
 def evaluate(a,s):
  nonlocal used
  if used>=424 or time.monotonic()>=deadline:raise Budget('EF/wall cap')
  used+=1;e,f=s.evaluate(a);raw=e
  for term in terms:be,bf=term.evaluate(a);e+=be;f+=bf
  return dict(request=used,energy=e,raw_energy=raw,max_force=float(np.linalg.norm(f,axis=1).max()),positions=a.positions.tolist(),forces=f.tolist())
 with (OUT/'evaluations.jsonl').open('w') as log:
  try:
   for _ in range(423):
    a.positions=oracle.x();current=evaluate(a,s);rows.append(current);log.write(json.dumps(current)+'\n');log.flush()
    if len(rows)==1 and current['max_force']<=.01:status='converged';break
    flag=oracle.advance(current['energy'],np.array(current['forces']),current)
    if oracle.stop_reason:status=oracle.stop_reason;break
    if flag!=1:status='native_converged_unqualified' if flag==0 else 'native_failure';break
  except Exception as exc:error=repr(exc);status='budget' if isinstance(exc,Budget) or time.monotonic()>=deadline else 'error'
  finally:
   dump(OUT/'pre-fresh.json',dict(status=status,error=error,physical_requests=used,accepted=oracle.accepted,calls=oracle.calls))
  try:
   endpoint=oracle.accepted[-1] if oracle.accepted else rows[0];a.positions=np.array(endpoint['positions']);fresh=evaluate(a,surface());fresh['energy_error']=fresh['energy']-endpoint['energy'];log.write(json.dumps(dict(fresh=True,**fresh))+'\n');log.flush()
  except Exception as exc:fresh=dict(error=repr(exc))
  finally:
   signal.setitimer(signal.ITIMER_REAL,0);dump(OUT/'result.json',dict(status=status,error=error,physical_requests=used,optimization_requests=len(rows),accepted_steps=len(oracle.accepted),accepted=oracle.accepted,calls=oracle.calls,native_flag=flag,fresh=fresh,seconds=time.monotonic()-start));print(json.dumps(dict(status=status,error=error,physical_requests=used,accepted_steps=len(oracle.accepted),calls=oracle.calls,fresh={k:v for k,v in fresh.items() if k not in ['positions','forces']},seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
