"""Bounded continuation of completed Gaussian8; research only, no defaults changed."""
import argparse, hashlib, json, math, shutil, time
from pathlib import Path
import numpy as np
from ase import Atoms, units
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart, ASEStressSurface
from pamssw.standalone.generalized_numerics import safe_lbfgs, generalized_dimer
from pamssw.standalone.cell_relax import relax_cell_coordinates
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step'
SRC=BASE/'joint-memory400-whole-step'
OUT=BASE/'joint-memory400-continuation-root'
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
def encode(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)
def save(name,obj): (OUT/name).write_text(json.dumps(obj,default=encode,indent=2,allow_nan=False)+'\n')
def atoms(d): return Atoms(**{k:d[k] for k in ('numbers','positions','cell','pbc')})
def atomdict(a): return dict(numbers=a.numbers,positions=a.positions,cell=a.cell.array,pbc=a.pbc)
def main(execute=False):
    OUT.mkdir(exist_ok=True)
    src=json.loads((SRC/'result.json').read_text()); rec=src['records'][1]; cfg=src['joint_config']
    assert rec['status']=='evaluation_failed' and rec['landing'] is None
    assert len(rec['frozen_gaussians'])==9 and all(c['index']==i and c['status']=='converged' for i,c in enumerate(rec['climb'][:9]))
    assert len(rec['climb'])==10 and 'index' not in rec['climb'][9]
    assert cfg['lbfgs_memory']==400 and src['seed']==3
    chart=SymmetricLogStrainChart(atoms(rec['chart_reference']),strain_length=cfg['strain_length'])
    work=np.array(rec['climb'][8]['q']); terms=[(np.array(t['center']),np.array(t['direction']),t['weight']) for t in rec['frozen_gaussians']]
    for i,t in enumerate(terms):
        if i: assert np.array_equal(t[0],rec['climb'][i-1]['q'])
    rng=np.random.default_rng(src['seed']); anchor=chart.project(rng.normal(size=work.size)); anchor/=np.linalg.norm(anchor)
    source_checks={}
    for name in ['vc_reference.py','vc_geometry.py','generalized_numerics.py','cell_relax.py']:
        rel=Path('pamssw/standalone')/name; assert (ROOT/rel).read_bytes()==(SRC/'source'/rel).read_bytes();source_checks[str(rel)]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    assert (ROOT/'pamssw/relax.py').read_bytes()==(SRC/'source/pamssw/relax.py').read_bytes()
    save('plan.json',dict(start='completed climb[8].q',pending_index=9,old_terms=9,max_new_EFS=1000,max_search_EFS=999,wall_seconds=600,config=cfg,source_checks=source_checks,source_result_sha256=hashlib.sha256((SRC/'result.json').read_bytes()).hexdigest(),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),rng_basis='source run_vc_ssw consumes only one normal(150) draw before first MC; state derived, not serialized',rng_after_anchor=rng.bit_generator.state,anchor=anchor,initial_q=work))
    if not execute: print('prepare-only validated, zero PES'); return
    assert not (OUT/'result.json').exists() and not (OUT/'ledger.jsonl').exists()
    shutil.copytree(SRC/'source',OUT/'source',dirs_exist_ok=True)
    shutil.copy2(__file__,OUT/'runner-executed.py')
    from mace.calculators import MACECalculator
    import ase,torch,mace,sys,os
    save('environment.json',dict(python=sys.executable,numpy=np.__version__,ase=ase.__version__,torch=torch.__version__,mace=mace.__version__,numpy_file=np.__file__,ase_file=ase.__file__,env={k:os.environ.get(k) for k in ['PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']}))
    start=time.monotonic(); role='rotation9'; ledger=[]
    class Surface(ASEStressSurface):
        def evaluate(self,a):
            if self.requests>=999 or time.monotonic()-start>=600: raise RuntimeError('declared continuation budget exhausted')
            row=dict(role=role,request=self.requests+1,q=chart.pack(a),atoms=atomdict(a)); before=self.requests
            try:
                e,f,s=super().evaluate(a); row.update(energy=e,forces=f,stress=s);return e,f,s
            except Exception as exc:
                row['error']=str(exc);raise
            finally:
                row['physical_requests']=self.requests-before;ledger.append(row)
                with (OUT/'ledger.jsonl').open('a') as fp: fp.write(json.dumps(row,default=encode,allow_nan=False)+'\n')
    surface=Surface(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64'))
    def physical(q): return chart.evaluate(q,surface.evaluate,pressure=cfg['pressure'])
    def bare(q):
        ev=physical(q);return ev.objective,chart.project(ev.gradient)
    def biased(q):
        e,g=bare(q)
        for c,n,w in terms:
            z=float((q-c)@n);b=w*math.exp(-.5*(z/cfg['width'])**2);e+=b;g-=b*z/cfg['width']**2*n
        return e,g
    def norm(v): return max(float(np.linalg.norm(v[:-6].reshape(-1,3),axis=1).max()),float(np.linalg.norm(v[-6:])))
    out=dict(status='running',rows=[],landing=None,accepted=False,fresh=None)
    try:
        mode=generalized_dimer(work,anchor,rotation_bias=cfg['rotation_bias'],fd_step=cfg['fd_step'],max_hvp=cfg['rotation_hvp'],tol=cfg['rotation_tol'],evaluate=bare)
        out['mode']=dict(direction=mode.direction,residual=mode.residual_norm,converged=mode.converged)
        if not mode.converged: raise RuntimeError('stage9 rotation failed')
        displaced=work+cfg['width']*mode.direction;role='height9';_,g=biased(displaced)
        weight=float((cfg['forward_force']+g@mode.direction)*cfg['width']*math.exp(.5))
        if not np.isfinite(weight) or weight<=0: raise RuntimeError('stage9 nonpositive height')
        terms.append((work.copy(),mode.direction.copy(),weight));role='biased_quench9'
        fit=safe_lbfgs(displaced,biased,gradient_norm=norm,step_norm=norm,gtol=cfg['gradient_tol'],max_step=cfg['max_step'],maxiter=cfg['relax_steps'],lbfgs_memory=400)
        work=fit.q.copy();out['rows'].append(dict(index=9,status=fit.status,q=work,weight=weight,requests=fit.requests,steps=fit.steps))
        if not fit.converged: raise RuntimeError('stage9 biased quench failed: '+fit.status)
        role='post_climb_true_energy';ev=physical(work);out['rows'][-1]['true_objective']=ev.objective
        role='true_cell_quench'
        landing=relax_cell_coordinates(chart,work,surface,pressure=cfg['pressure'],fmax=cfg['fmax'],stress_tol=cfg['stress_tol'],max_step=cfg['max_step'],maxiter=cfg['relax_steps'],lbfgs_memory=400)
        work=landing.q.copy();role='landing_certificate';ev=physical(work)
        fmax=float(np.linalg.norm(ev.forces,axis=1).max());smax=float(np.abs(ev.stress+cfg['pressure']*np.eye(3)).max());cert=landing.converged and fmax<=cfg['fmax'] and smax<=cfg['stress_tol']
        delta=ev.objective-src['current']['objective'];out['landing']=dict(atoms=atomdict(ev.atoms),objective=ev.objective,delta=delta,fmax=fmax,stress_max=smax,volume=ev.volume,certified=cert,optimizer_status=landing.status,optimizer_requests=landing.requests)
        out['status']='completed' if cert else 'true_quench_failed'
        if cert:
            out['accepted']=bool(delta<=0 or (cfg['temperature_K']>0 and rng.random()<math.exp(-delta/(units.kB*cfg['temperature_K']))))
            fresh=ASEStressSurface(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64'))
            ee,ff,ss=fresh.evaluate(ev.atoms)
            out['fresh']=dict(requests=fresh.requests,energy=ee,forces=ff,stress=ss,fmax=float(np.linalg.norm(ff,axis=1).max()),stress_max=float(np.abs(ss+cfg['pressure']*np.eye(3)).max()),objective=ee+cfg['pressure']*ev.volume)
    except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        out['status']='failed';out['error']=str(exc);out['failed_role']=role
    finally:
        out.update(last_q=work,terms=[dict(center=c,direction=n,weight=w,width=cfg['width']) for c,n,w in terms],search_requests=surface.requests,fresh_requests=0 if out['fresh'] is None else out['fresh']['requests'],wall_seconds=time.monotonic()-start,rng_state=rng.bit_generator.state)
        out['total_new_EFS']=out['search_requests']+out['fresh_requests'];assert out['total_new_EFS']<=1000;assert sum(r['physical_requests'] for r in ledger)==surface.requests
        save('result.json',out);print(json.dumps({k:out[k] for k in ['status','search_requests','fresh_requests','total_new_EFS','wall_seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)
