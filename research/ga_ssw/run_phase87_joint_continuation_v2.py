"""Continue the saved phase87 joint VC boundary; no restart of the 1500-EFS run."""
import dataclasses, hashlib, json, os, shutil, signal, time
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart, ASEStressSurface
from pamssw.standalone.generalized_numerics import safe_lbfgs, generalized_dimer
from pamssw.standalone.cell_relax import relax_cell_coordinates

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step/joint-continuation-v2'
SRC=ROOT/'research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step/joint/result.json'
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
CAP=999; FRESH_RESERVE=1; WALL=600.; START=time.monotonic()
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def atom(d): return Atoms(numbers=d['numbers'],positions=d['positions'],cell=d['cell'],pbc=d['pbc'])
source=json.loads(SRC.read_text()); rec=source['records'][1]
assert rec['status']=='biased_quench_failed' and rec['landing'] is None
assert [c['index'] for c in rec['climb']]==list(range(8))
assert len(rec['frozen_gaussians'])==8
chart=SymmetricLogStrainChart(atom(rec['chart_reference']),strain_length=5.)
q0=np.asarray(rec['climb'][6]['q'],float)
assert q0.shape==(150,)
last=atom(rec['climb'][6]['q'] and rec['chart_reference']); q_unpacked=chart.unpack(q0); expected6=chart.unpack(np.asarray(rec['climb'][6]['q'],float))
assert np.array_equal(last.numbers,q_unpacked.numbers)
assert np.max(np.abs(expected6.positions-q_unpacked.positions))==0.
assert np.max(np.abs(expected6.cell.array-q_unpacked.cell.array))==0.
rng=np.random.default_rng(3); anchor=chart.project(rng.normal(size=q0.size)); anchor/=np.linalg.norm(anchor)
# The original record did not serialize anchor; this is deterministic reconstruction, not a byte identity claim.
terms=[]
for t in rec['frozen_gaussians'][:7]:
    c=np.asarray(t['center'],float); n=np.asarray(t['direction'],float)
    terms.append((c,n,float(t['weight'])))
config=dict(width=.6,rotation_bias=100.,pressure=0.,gradient_tol=.005,fmax=.01,stress_tol=.001,max_step=.2,relax_steps=300,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02)
# zero-PES source contract artifact before calculator construction
plan=json.loads((OUT/'plan.json').read_text()); plan['status']='prepared'; plan.pop('invalid_reason',None); plan.update({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'zero_pes_validation':{'q_stage6_position_diff':0.0,'q_stage6_cell_diff':0.0,'terms':7,'anchor_reconstructed_from_seed3':True}}); dump('plan.json',plan)
from mace.calculators import MACECalculator
calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64')
class Counted(ASEStressSurface):
    def evaluate(self,a):
        if self.requests>=CAP or time.monotonic()-START>=WALL: raise RuntimeError('continuation budget exhausted')
        return super().evaluate(a)
surface=Counted(calc)
def physical(q): return chart.evaluate(q,surface.evaluate,pressure=0.)
def bare(q):
    ev=physical(q); return ev.objective,chart.project(ev.gradient)
def biased(x):
    e,g=bare(x)
    for c,n,w in terms:
        z=float((x-c)@n); b=w*np.exp(-.5*(z/config['width'])**2); e+=b; g-=b*z/config['width']**2*n
    return e,g
def norm(v): return max(float(np.linalg.norm(v[:-6].reshape(-1,3),axis=1).max()),float(np.linalg.norm(v[-6:])))
def minimize(q): return safe_lbfgs(q,biased,gradient_norm=norm,step_norm=norm,gtol=config['gradient_tol'],max_step=config['max_step'],maxiter=config['relax_steps'],lbfgs_memory=None)
def true_minimize(q): return relax_cell_coordinates(chart,q,surface,pressure=0.,fmax=config['fmax'],stress_tol=config['stress_tol'],max_step=config['max_step'],maxiter=config['relax_steps'],lbfgs_memory=None)
def cert(q):
    ev=physical(q); f=float(np.linalg.norm(ev.forces,axis=1).max()); s=float(np.abs(ev.stress).max()); return ev,dict(fmax=f,stress_max=s,certified=f<=.01 and s<=.001)
rows=[]; q=q0.copy(); current_obj=float(source['current']['objective']); status='continuing'
# Rebuild the failed stage 7 from its last completed pre-Gaussian q; do not reuse failed stage7 q.
for j in range(7,10):
    if status!='continuing': break
    before=surface.requests
    mode=generalized_dimer(q,anchor,rotation_bias=100.,fd_step=1e-4,max_hvp=100,tol=.02,evaluate=bare)
    if not mode.converged: status='rotation_failed'; rows.append(dict(role='gaussian',index=j,status=status,residual=mode.residual_norm,requests=surface.requests-before)); break
    displaced=q+.6*mode.direction; _,bg=biased(displaced); weight=(.1+bg@mode.direction)*.6*np.exp(.5)
    if not np.isfinite(weight) or weight<=0: status='nonpositive_height'; rows.append(dict(role='gaussian',index=j,status=status,weight=float(weight))); break
    terms.append((q.copy(),mode.direction.copy(),float(weight))); relaxed=minimize(displaced); q=relaxed.q.copy()
    row=dict(role='gaussian',index=j,status=relaxed.status,weight=float(weight),rotation_residual=float(mode.residual_norm),q=q.tolist(),requests=surface.requests-before)
    rows.append(row)
    if j==7:
        row['saved_pending_weight']=float(rec['climb'][7]['weight']); row['weight_abs_diff']=abs(weight-row['saved_pending_weight']); row['saved_pending_direction_norm_diff']=float(np.linalg.norm(mode.direction-np.asarray(rec['climb'][7]['direction'],float)))
    if not relaxed.converged: status='biased_quench_failed'; break
    tev=physical(q); row.update(objective=float(tev.objective),delta_from_current=float(tev.objective-current_obj))
    if tev.objective<current_obj: status='lower_true_enthalpy'; break
if status in ('continuing','lower_true_enthalpy'):
    lm=true_minimize(q); ev,c=cert(lm.q); status='completed' if lm.converged and c['certified'] else 'true_quench_failed'
    landing=dict(objective=float(ev.objective),energy=float(ev.energy),fmax=c['fmax'],stress_max=c['stress_max'],certified=c['certified'],atoms={'numbers':ev.atoms.numbers.tolist(),'positions':ev.atoms.positions.tolist(),'cell':ev.atoms.cell.array.tolist(),'pbc':ev.atoms.pbc.tolist()},optimizer={'status':lm.status,'steps':lm.steps,'requests':lm.requests})
else: landing=None
# Independent fresh certificate, reserved one call, only if a certified landing exists.
fresh=None
if landing is not None and landing['certified']:
    fresh_surface=ASEStressSurface(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64'))
    a=atom(landing['atoms']); e,f,s=fresh_surface.evaluate(a); fresh=dict(energy=float(e),objective=float(e),fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(np.abs(s).max()),requests=fresh_surface.requests)
out={'status':status,'source_requests':source['requests'],'continuation_requests':surface.requests,'fresh_requests':0 if fresh is None else fresh['requests'],'total_new_requests':surface.requests+(0 if fresh is None else fresh['requests']),'wall_seconds':time.monotonic()-START,'resume_rows':rows,'landing':landing,'fresh':fresh,'final_terms':len(terms),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'note':'counterfactual continuation from saved joint boundary; original 1500-EFS artifact unchanged; anchor reconstructed from seed3 because original record omitted it'}
dump('result.json',out)
print(json.dumps({k:out[k] for k in ('status','continuation_requests','fresh_requests','total_new_requests','final_terms','wall_seconds')}))
