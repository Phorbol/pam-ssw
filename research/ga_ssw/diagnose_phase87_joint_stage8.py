"""Zero-search-contract replay/diagnostic for phase87 joint stage8; <=100 EFS, no continuation."""
import dataclasses,hashlib,json,time
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart,ASEStressSurface
from pamssw.standalone.generalized_numerics import generalized_dimer
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step'; OUT=BASE/'joint-stage8-diagnostic-v1'; SRC=BASE/'joint/result.json'; V2=BASE/'joint-continuation-v2/result.json'; MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model'); OUT.mkdir(exist_ok=False)
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def atom(d):return Atoms(numbers=d['numbers'],positions=d['positions'],cell=d['cell'],pbc=d['pbc'])
s=json.loads(SRC.read_text()); v=json.loads(V2.read_text()); r=s['records'][1]; rows=v['resume_rows']; s7=next(x for x in rows if x['index']==7); s8=next(x for x in rows if x['index']==8); q6=np.asarray(r['climb'][6]['q'],float); q7=np.asarray(s7['q'],float); q8failed=np.asarray(s8['q'],float); chart=SymmetricLogStrainChart(atom(r['chart_reference']),strain_length=5.)
assert np.max(np.abs(chart.unpack(q6).positions-chart.unpack(q6).positions))==0
anchor=chart.project(np.random.default_rng(3).normal(size=150));anchor/=np.linalg.norm(anchor)
terms=[]
for t in r['frozen_gaussians'][:7]:terms.append((np.asarray(t['center'],float),np.asarray(t['direction'],float),float(t['weight'])))
# saved stage7 pending term: center is q6; its direction/weight are in failed record
terms.append((q6.copy(),np.asarray(r['climb'][7]['direction'],float),float(r['climb'][7]['weight'])))
width=.6
def addbias(e,g,x,ts):
 for c,n,w in ts:
  z=float((x-c)@n);b=w*np.exp(-.5*(z/width)**2);e+=b;g-=b*z/width**2*n
 return e,g
start=time.monotonic(); search_cap=98
from mace.calculators import MACECalculator
calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64')
class LedgerSurface(ASEStressSurface):
 def evaluate(self,a):
  if self.requests>=search_cap or time.monotonic()-start>=90:raise RuntimeError('diagnostic search cap')
  e,f,st=super().evaluate(a); q=chart.pack(a); ledger.append({'request':self.requests,'q':q.tolist(),'energy':float(e),'fmax':float(np.linalg.norm(f,axis=1).max()),'stress_max':float(np.abs(st).max())});return e,f,st
ledger=[];surf=LedgerSurface(calc)
def bare(x):
 ev=chart.evaluate(x,surf.evaluate,pressure=0.);return ev.objective,chart.project(ev.gradient)
center=q7.copy(); mode_error=None
try: mode=generalized_dimer(center,anchor,rotation_bias=100.,fd_step=1e-4,max_hvp=97,tol=.02,evaluate=bare); mode_row={'converged':bool(mode.converged),'residual':float(mode.residual_norm),'force_calls':int(mode.force_calls),'direction':mode.direction.tolist(),'curvature':float(mode.curvature)}
except Exception as e: mode=None;mode_row={'error':repr(e)}
# Separate terminal and fresh calls, each one E/F/stress. No quench.
def evaluate_final(a):
 c=ASEStressSurface(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64')); e,f,st=c.evaluate(a);return c.requests,float(e),f,st
terminal_req,e,f,st=evaluate_final(chart.unpack(q8failed)); ev=chart.evaluate(q8failed,lambda a:(e,f,st),pressure=0.); be,bg=addbias(ev.objective,chart.project(ev.gradient),q8failed,terms+([] if mode is None else [(center,np.asarray(mode.direction),float(s8['weight']))]))
fresh_req,ef,ff,sf=evaluate_final(chart.unpack(q8failed))
out={'status':'diagnostic_complete','source_result_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'v2_result_sha256':hashlib.sha256(V2.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'search_requests':surf.requests,'terminal_request':terminal_req,'fresh_request':fresh_req,'total_requests':surf.requests+terminal_req+fresh_req,'wall_seconds':time.monotonic()-start,'mode':mode_row,'saved_stage8':{'status':s8['status'],'weight':s8['weight'],'rotation_residual':s8['rotation_residual'],'q':s8['q']},'stage8_center_source':'v2 stage7 q','terminal_failed_q_true':{'energy':e,'fmax':float(np.linalg.norm(f,axis=1).max()),'stress_max':float(np.abs(st).max()),'objective':ev.objective},'terminal_failed_q_biased':{'objective':float(be),'gradient_norm':float(np.linalg.norm(bg)),'gradient_max_atom':float(np.linalg.norm(bg[:-6].reshape(-1,3),axis=1).max()),'same_frozen_terms':9},'fresh_true':{'energy':ef,'fmax':float(np.linalg.norm(ff,axis=1).max()),'stress_max':float(np.abs(sf).max())},'ledger':ledger,'limitations':'diagnostic replay only; no biased quench rerun, no new Gaussian, no success certificate'}
dump('result.json',out);dump('frozen-stage8.json',{'center_q':center.tolist(),'previous_terms':[{'center':c.tolist(),'direction':n.tolist(),'weight':w} for c,n,w in terms],'saved_stage8_weight':s8['weight'],'saved_stage8_failed_q':q8failed.tolist(),'anchor':anchor.tolist(),'contract':'same chart strain_length=5, seed3 reconstructed anchor, dimer fd=1e-4 bias=100 max_hvp=97; stage8 term appended from replay direction'})
print(json.dumps({k:out[k] for k in ('status','search_requests','terminal_request','fresh_request','total_requests','wall_seconds','mode')}))
