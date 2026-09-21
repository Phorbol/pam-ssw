"""Prepare/run Safe-total memory10 vs memory400 on the reconstructed phase87 stage8 objective.

Preparation is inert unless --execute is supplied.  Both arms use the same frozen
stage8 center, nine terms, chart, seed-independent anchor, and MACE calculator
settings.  Every physical E/F/stress request is recorded, including rejected
line-search trials; fresh validation is recorded separately.
"""
import argparse, hashlib, json, os, time, shutil, sys
from pathlib import Path
import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step'
FROZEN = BASE/'joint-stage8-diagnostic-v1/frozen-stage8.json'
SOURCE = BASE/'joint/result.json'
DIAG = BASE/'joint-stage8-diagnostic-v1/result.json'
MODEL = Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
OUT = BASE/'joint-stage8-memory-compare-v1'
MAX_EFS, WALL, FRESH_RESERVE = 500, 300.0, 1

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def atom(d): return Atoms(numbers=d['numbers'], positions=d['positions'], cell=d['cell'], pbc=d['pbc'])

def main(execute=False):
    frozen=json.loads(FROZEN.read_text()); source=json.loads(SOURCE.read_text()); diag=json.loads(DIAG.read_text())
    assert frozen['term_count']==9 and len(frozen['previous_terms'])==9
    assert len(frozen['center_q'])==150
    assert diag['mode']['residual'] < .02
    assert frozen['saved_stage8_weight_reused'] is True
    reference=atom(source['records'][1]['chart_reference'])
    dep={p:sha(ROOT/p) for p in ('pamssw/standalone/vc_geometry.py','pamssw/standalone/generalized_numerics.py','pamssw/standalone/cell_relax.py','pamssw/relax.py')}
    snap=BASE/'joint/source'
    equal={p:(ROOT/p).read_bytes()==(snap/p).read_bytes() for p in dep}
    assert all(equal.values())
    plan={'status':'prepared','execute_guard':'requires --execute; this preparation performed zero PES calls',
      'objective':'frozen reconstructed stage8 objective; not exact original trajectory',
      'source_result_sha256':sha(SOURCE),'diagnostic_result_sha256':sha(DIAG),'frozen_sha256':sha(FROZEN),'model_sha256':sha(MODEL),
      'dependencies_sha256':dep,'source_dependency_byte_equal_to_joint_snapshot':equal,
      'environment_required':{'python':'/home/gengjianrui/.conda/envs/mace_env/bin/python','PYTHONNOUSERSITE':'1','numpy':'2.0.2','ase':'3.26.0','torch':'2.8.0+cu128','mace_torch':'0.3.16','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''},
      'runtime_assertions':['imported ASE/NumPy versions and module __file__ under mace_env before any execute PES','all four dependency bytes compared to joint/source snapshot before execute'],
      'objective_contract':{'center_source':str(FROZEN),'terms':9,'width':.6,'rotation_bias':100.,'anchor_source':'frozen-stage8.json reconstructed seed3 anchor','stage8_direction':'fresh same-seed diagnostic reconstruction','stage8_weight':'saved v2 weight reused; not independently re-derived','chart':'SymmetricLogStrainChart strain_length=5; pressure=0'},
      'arms':[{'lbfgs_memory':10,'max_requests':500,'wall_seconds':300},{'lbfgs_memory':400,'max_requests':500,'wall_seconds':300}],
      'optimizer':{'maxiter':300,'max_step':.2,'gtol':.005,'fmax':.01,'stress_tol':.001},
      'accounting':{'preflight_fresh':1,'search_cap_per_arm':500,'maximum_new_efs':1001,'original_phase87_and_prior_continuations_excluded':True},
      'ledger_contract':['request index','role','arm','q','energy','forces','stress','objective','gradient','accepted/rejected trial','secant accepted/rejected','status'],
      'limitations':['reconstructed objective due missing original stage8 direction','same frozen start; no Gaussian or L changes','not a full SS-W success test']}
    OUT.mkdir(exist_ok=True)
    dump(OUT/'plan.json',plan)
    if not execute:
        print(json.dumps({'status':'prepared','path':str(OUT/'plan.json'),'pes_calls':0}))
        return
    if (OUT/'result.json').exists() or list(OUT.glob('ledger-memory*.jsonl')):
        raise RuntimeError('execution artifacts already exist; preserve prior attempt')
    shutil.copy2(__file__, OUT/'runner-executed.py')
    shutil.copytree(ROOT/'pamssw',OUT/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(OUT/'source'))
    dump(OUT/'frozen-input.json',frozen)
    # Actual execution is intentionally explicit and bounded.
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart, ASEStressSurface
    from pamssw.standalone.generalized_numerics import safe_lbfgs
    from pamssw.standalone.cell_relax import relax_cell_coordinates
    from mace.calculators import MACECalculator
    import ase, numpy, torch, mace
    assert numpy.__version__=='2.0.2' and ase.__version__=='3.26.0'
    assert str(Path(ase.__file__).resolve()).startswith('/home/gengjianrui/.conda/envs/mace_env/')
    assert str(Path(numpy.__file__).resolve()).startswith('/home/gengjianrui/.conda/envs/mace_env/')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dump(OUT/'runtime-environment.json',dict(python=sys.executable,
      modules={m.__name__:dict(version=str(getattr(m,'__version__','unknown')),path=m.__file__) for m in (ase,numpy,torch,mace)},
      environment={k:os.environ.get(k) for k in ('PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')}))
    chart=SymmetricLogStrainChart(reference,strain_length=5.)
    center_q=np.asarray(frozen['center_q'],float)
    stage8=frozen['previous_terms'][-1]
    q0=center_q + .6*np.asarray(stage8['direction'],float)
    anchor=np.asarray(frozen['anchor'],float)
    terms=[(np.asarray(t['center']),np.asarray(t['direction']),float(t['weight'])) for t in frozen['previous_terms']]
    config={'width':.6,'gradient_tol':.005,'fmax':.01,'stress_tol':.001,'max_step':.2,'maxiter':300,'pressure':0.}
    class LedgerSurface(ASEStressSurface):
        def __init__(self, calc, arm): super().__init__(calc); self.arm=arm; self.ledger=[]; self.started=time.monotonic()
        def evaluate(self, a, role='optimizer_eval', extra=None):
            if self.requests>=MAX_EFS or time.monotonic()-self.started>=WALL: raise RuntimeError('bounded arm budget exhausted')
            e,f,s=super().evaluate(a)
            row={'request':self.requests-1,'arm':self.arm,'role':role,'q':chart.pack(a).tolist(),'energy':float(e),'forces':np.asarray(f).tolist(),'stress':np.asarray(s).tolist()}
            if extra: row.update(extra)
            self.ledger.append(row); return e,f,s
    # One separately accounted fresh displaced evaluation verifies the reconstructed
    # stage8 Gaussian height before either optimizer arm.  It is not an arm result.
    pre_calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64')
    pre_surface=ASEStressSurface(pre_calc)
    q_disp=q0
    pre_ev=chart.evaluate(q_disp,pre_surface.evaluate,pressure=0.)
    # Recompute only the proposed stage8 height from the displaced-point gradient;
    # do not alter the saved weight.  This is one independent preflight EFS.
    g=chart.project(pre_ev.gradient)
    for t in frozen['previous_terms'][:-1]:
        c=np.asarray(t['center']); n=np.asarray(t['direction']); w=float(t['weight']); z=float((q_disp-c)@n)
        b=w*np.exp(-.5*(z/.6)**2); g-=b*z/.6**2*n
    n=np.asarray(stage8['direction'],float)
    reconstructed_height=float((.1+g@n)*.6*np.exp(.5))
    preflight={'requests':pre_surface.requests,'q':q_disp.tolist(),'energy':float(pre_ev.energy),'objective':float(pre_ev.objective),'forces':np.asarray(pre_ev.forces).tolist(),'stress':np.asarray(pre_ev.stress).tolist(),'reconstructed_height':reconstructed_height,'saved_weight':float(stage8['weight']),'height_abs_diff':abs(reconstructed_height-float(stage8['weight']))}
    preflight['height_tolerance_eV']=1e-6  # equality check only, not a fitted search parameter
    dump(OUT/'preflight.json',preflight)
    if preflight['height_abs_diff']>preflight['height_tolerance_eV']:
        dump(OUT/'result.json',dict(status='preflight_mismatch',preflight=preflight,total_requests=pre_surface.requests))
        return
    def run_arm(memory):
        surface=LedgerSurface(MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64'),memory)
        def physical(q,role='physical',extra=None): return chart.evaluate(q,lambda a: surface.evaluate(a,role,extra),pressure=0.)
        def bare(q,role='physical',extra=None):
            ev=physical(q,role,extra); return ev.objective,chart.project(ev.gradient)
        def biased(q,role='physical',extra=None):
            e,g=bare(q,role,extra)
            for c,n,w in terms:
                z=float((q-c)@n); b=w*np.exp(-.5*(z/config['width'])**2); e+=b; g-=b*z/config['width']**2*n
            if surface.ledger:
                surface.ledger[-1].update({'biased_objective':float(e),'biased_gradient':g.tolist()})
                with (OUT/f'ledger-memory{memory}.jsonl').open('a') as handle:
                    handle.write(json.dumps(surface.ledger[-1],allow_nan=False)+'\n')
            return e,g
        def norm(g): return max(float(np.linalg.norm(g[:-6].reshape(-1,3),axis=1).max()),float(np.linalg.norm(g[-6:])))
        res=safe_lbfgs(q0,lambda q: biased(q,'optimizer_eval'),gradient_norm=norm,step_norm=norm,gtol=.005,max_step=.2,maxiter=300,max_requests=MAX_EFS,lbfgs_memory=memory)
        accepted={t['requests']-1 for t in res.trace}
        for row in surface.ledger:
            row['accepted_point']=row['request'] in accepted
        report={'memory':memory,'status':res.status,'requests':surface.requests,
          'optimizer_attempts':res.requests,'wall_seconds':time.monotonic()-surface.started,'steps':res.steps,'accepted_secants':res.accepted_secants,'rejected_secants':res.rejected_secants,'rejected_trials':res.rejected_trials,'trace':[{'q':x['q'].tolist(),'energy':x['energy'],'gradient':x['gradient'].tolist(),'gradient_norm':x['gradient_norm'],'requests':x['requests'],'step':x['step']} for x in res.trace],'ledger':surface.ledger}
        dump(OUT/f'result-memory{memory}.json',report)
        return report
    # Preflight fresh and arms would be recorded here; execution is outside preparation.
    results=[run_arm(m) for m in (10,400)]
    dump(OUT/'result.json',{'status':'completed','arms':results,'total_requests':pre_surface.requests+sum(r['requests'] for r in results),'preflight':preflight,'preflight_requests':pre_surface.requests,'source_result_sha256':sha(SOURCE),'frozen_sha256':sha(FROZEN),'model_sha256':sha(MODEL)})
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); main(ap.parse_args().execute)
