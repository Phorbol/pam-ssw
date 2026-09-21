"""Bounded central directional derivative check for the frozen reconstructed stage8 objective."""
import json, hashlib, time
from pathlib import Path
import numpy as np
from ase import Atoms
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step'; OUT=BASE/'joint-stage8-memory-compare-v1'; FROZEN=OUT/'frozen-input.json'; SOURCE=BASE/'joint/result.json'; MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atom(d): return Atoms(numbers=d['numbers'],positions=d['positions'],cell=d['cell'],pbc=d['pbc'])
def main():
 plan={'status':'prepared','pes_calls':0,'source':str(FROZEN),'directions':['atomic_translation_projected','strain6','mixed_projected'],'steps':[1e-4,5e-5],'expected_requests':13,'cap':15,'wall_seconds':90,'seed':20260911,'purpose':'analytic directional derivative vs central FD; not physical success'}
 (OUT/'directional-fd-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 import ase,numpy,torch,mace
 assert ase.__version__=='3.26.0' and numpy.__version__=='2.0.2'
 assert str(Path(ase.__file__).resolve()).startswith('/home/gengjianrui/.conda/envs/mace_env/')
 from mace.calculators import MACECalculator
 from pamssw.standalone.vc_geometry import SymmetricLogStrainChart, ASEStressSurface
 frozen=json.loads(FROZEN.read_text()); source=json.loads(SOURCE.read_text()); ref=atom(source['records'][1]['chart_reference']); chart=SymmetricLogStrainChart(ref,strain_length=5.)
 center=np.asarray(frozen['center_q'],float); terms=[(np.asarray(t['center']),np.asarray(t['direction']),float(t['weight'])) for t in frozen['previous_terms']]; q0=center+.6*terms[-1][1]
 surface=ASEStressSurface(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64')); ledger=[]
 def evalq(q,role):
  ev=chart.evaluate(q,surface.evaluate,pressure=0.); g=chart.project(ev.gradient); e=float(ev.objective)
  for c,n,w in terms:
   z=float((q-c)@n); b=w*np.exp(-.5*(z/.6)**2); e+=b; g-=b*z/.6**2*n
  row={'request':surface.requests-1,'role':role,'q':q.tolist(),'objective':e,'analytic_gradient':g.tolist(),'energy':float(ev.energy),'forces':np.asarray(ev.forces).tolist(),'stress':np.asarray(ev.stress).tolist()}; ledger.append(row); return e,g
 e0,g0=evalq(q0,'center'); rng=np.random.default_rng(20260911); dirs=[]
 a=np.zeros(150); a[:3]=1.; dirs.append(('atomic_translation_projected',chart.project(a)))
 b=np.zeros(150); b[-6]=1.; dirs.append(('strain6',chart.project(b)))
 c=chart.project(rng.normal(size=150)); dirs.append(('mixed_projected',c))
 rows=[]
 for name,d in dirs:
  d=d/np.linalg.norm(d); ad=float(g0@d)
  for h in (1e-4,5e-5):
   ep,_=evalq(q0+h*d,name+'_plus'); em,_=evalq(q0-h*d,name+'_minus'); fd=(ep-em)/(2*h); rows.append({'direction':name,'h':h,'analytic_directional_derivative':ad,'central_fd':fd,'absolute_error':abs(fd-ad),'relative_error':abs(fd-ad)/max(1.,abs(ad))})
 out={'status':'complete','requests':surface.requests,'wall_seconds':time.monotonic()-plan_start,'source_sha256':sha(SOURCE),'frozen_sha256':sha(FROZEN),'model_sha256':sha(MODEL),'environment':{'python':str(Path(__import__('sys').executable).resolve()),'numpy':numpy.__version__,'ase':ase.__version__,'torch':torch.__version__,'mace':mace.__version__,'ase_file':ase.__file__,'numpy_file':numpy.__file__},'center':{'q':q0.tolist(),'objective':e0,'analytic_gradient':g0.tolist()},'directions':rows,'ledger':ledger,'note':'directional derivative check only; no optimization or physical qualification'}
 (OUT/'directional-fd-result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':out['status'],'requests':out['requests'],'max_abs_error':max(x['absolute_error'] for x in rows),'wall_seconds':out['wall_seconds']}))
plan_start=time.monotonic()
if __name__=='__main__': main()
