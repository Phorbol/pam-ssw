"""Post-gate physical qualification; refuses execution before all eight finish."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.vc_geometry import ASEStressSurface,SymmetricLogStrainChart
from pamssw.standalone.generalized_numerics import safe_lbfgs
from research.ga_ssw.compare_vc_arms import serial

PLAN=dict(run_denominator=8,total_requests=6000,total_seconds=3600,threads=1,
    fmax=1e-4,stress_tol=1e-5,max_step=.2,relax_steps=100,relax_requests=101,
    hessian_steps=[1e-4,5e-5],phases='all fresh first; then each refine+hessian in endpoint-index/arm/system round robin',
    dedup='exact atoms/cell/PBC bytes; fresh shared across domains, refinement/Hessian only within same domain/pressure',
    source='qualify_vc_cu4_minima: physical stopping + own-chart 2-step Hessian; no universal tolerances claim')
class Limit(RuntimeError):pass


def collect(gate):
    manifest=json.loads((gate/'manifest.json').read_text());execution=json.loads((gate/'execution.json').read_text())
    commands=manifest['commands'];processes={r['name']:r for r in execution['runs']}
    if len(commands)!=8 or execution.get('status')=='running':raise ValueError('all eight gate runs must finish before qualification')
    runs=[];endpoints=[]
    for command in commands:
        name=Path(command[command.index('--output')+1]).name
        if name not in processes or 'returncode' not in processes[name]:raise ValueError('gate process still pending: '+name)
        path=gate/'results'/name/'result.json';arm=command[command.index('--arm')+1]
        system=Path(command[command.index('--input')+1]).stem
        row=dict(name=name,arm=arm,system=system,process=processes[name],result_path=str(path),source_status='missing',declared_model=command[command.index('--model')+1])
        runs.append(row)
        source_plan=path.parent/'plan.json'
        if source_plan.exists():row['source_model_sha256']=json.loads(source_plan.read_text()).get('model_sha256')
        if not path.exists():continue
        data=json.loads(path.read_text());row.update(source_status=data['status'],source_requests=data['requests'],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for landing in data['landings']:
            endpoints.append(dict(run=name,arm=arm,system=system,index=landing['index'],accepted=landing['accepted'],
                atoms=Atoms(**landing['atoms']),source_energy=landing['energy'],pressure=data['pressure'],
                domain='fixed' if arm=='fixed' else 'variable',strain_length=data['joint_config']['strain_length'],
                source_certificate=landing['certificate'],source_fmax=data['atomic_config']['fmax'],
                source_stress_tol=data['joint_config']['stress_tol']))
    order={arm:i for i,arm in enumerate(('fixed','pqc','block','joint'))}
    endpoints.sort(key=lambda e:(e['index'],order[e['arm']],e['system'],e['run']))
    return runs,endpoints


def key(a):return a.numbers.tobytes()+a.positions.tobytes()+a.cell.array.tobytes()+a.pbc.tobytes()


def qualify(runs,endpoints,surface,*,persist=lambda report:None,matching=None,plan=PLAN):
    report=dict(plan=plan,runs=runs,endpoints=[],tasks=[],requests=0,status='running')
    begin=surface.requests;cache={};tasks={}
    def save():report['requests']=surface.requests-begin;persist(report)
    # Phase 1 never spends a request on refinement before every original endpoint
    # has received a fresh check (or a recorded budget/error pending status).
    for e in endpoints:
        row=dict(e);row['fresh_status']='pending';report['endpoints'].append(row);k=key(e['atoms'])
        if k not in cache:
            try:
                energy,forces,stress=surface.evaluate(e['atoms'])
                cache[k]=dict(energy=energy,forces=forces,stress=stress,request=surface.requests-begin)
            except Exception as exc:cache[k]=dict(error=repr(exc),request=surface.requests-begin)
        fresh=cache[k]
        if 'error' in fresh:
            row.update(fresh_status='pending',error=fresh['error'],shared_request=fresh['request']);save();continue
        row.update(fresh_status='checked',fresh=fresh,
            energy_error=fresh['energy']-e['source_energy'],
            fresh_fmax=float(np.linalg.norm(fresh['forces'],axis=1).max()),
            fresh_stress_residual=float(abs(fresh['stress']+e['pressure']*np.eye(3)).max()))
        row['source_tolerance_force_pass']=row['fresh_fmax']<=e['source_fmax']
        row['source_tolerance_stress_pass']=row['fresh_stress_residual']<=e['source_stress_tol']
        row['source_domain_certificate_pass']=row['source_tolerance_force_pass'] and (e['domain']=='fixed' or row['source_tolerance_stress_pass'])
        taskkey=(k,e['domain'],e['pressure'],e['strain_length'])
        if taskkey not in tasks:
            task=dict(id=len(tasks),atoms=e['atoms'].copy(),domain=e['domain'],pressure=e['pressure'],
                strain_length=e['strain_length'],aliases=[],status='pending_refinement')
            tasks[taskkey]=task;report['tasks'].append(task)
        tasks[taskkey]['aliases'].append(dict(run=e['run'],index=e['index'],accepted=e['accepted']))
        row['task_id']=tasks[taskkey]['id'];save()
    for task in report['tasks']:
        if any(e['fresh_status']!='checked' for e in report['endpoints']):
            task['status']='pending_all_fresh_incomplete';save();continue
        a=task['atoms'];variable=task['domain']=='variable';p=task['pressure'];before=surface.requests
        chart=SymmetricLogStrainChart(a,strain_length=task['strain_length']) if variable else None
        q=chart.pack(a) if variable else a.positions.ravel().copy();conv={}
        def evaluate(x):
            if variable:
                ev=chart.evaluate(x,surface.evaluate,pressure=p);f=ev.forces;s=ev.stress;energy=ev.objective;g=chart.project(ev.gradient)
            else:
                b=a.copy();b.positions=x.reshape(-1,3);energy,f,s=surface.evaluate(b);g=-f.ravel()
            conv[x.tobytes()]=max(float(np.linalg.norm(f,axis=1).max())/plan['fmax'],
                float(abs(s+p*np.eye(3)).max())/plan['stress_tol'] if variable else 0.)
            return energy,g
        def norm(x):
            return max(float(np.linalg.norm(x[:-6].reshape(-1,3),axis=1).max()),float(np.linalg.norm(x[-6:]))) if variable else float(np.linalg.norm(x.reshape(-1,3),axis=1).max())
        try:
            relaxed=safe_lbfgs(q,evaluate,gradient_norm=norm,step_norm=norm,convergence_norm=lambda x,g:conv[x.tobytes()],gtol=1.,
                max_step=plan['max_step'],maxiter=plan['relax_steps'],max_requests=plan['relax_requests'])
            b=chart.unpack(relaxed.q) if variable else a.copy()
            if not variable:b.positions=relaxed.q.reshape(-1,3)
            task.update(refinement=relaxed,refined_atoms=b,refinement_requests=surface.requests-before,
                geometry_change=dict(max_same_order_displacement=float(np.linalg.norm(b.positions-a.positions,axis=1).max()),
                    cell_frobenius=float(np.linalg.norm(b.cell.array-a.cell.array)),volume_ratio=b.get_volume()/a.get_volume()),
                identity_comparison=matching(a,b) if matching is not None else 'pending independent structural matcher')
            if not relaxed.converged:
                task.update(status='pending_refinement_budget' if relaxed.status=='request_limit' or (relaxed.error and 'Limit:' in relaxed.error) else 'refinement_failed',requests=surface.requests-before);save();continue
            en,forces,stress=surface.evaluate(b)
            refined_fmax=float(np.linalg.norm(forces,axis=1).max());refined_stress=float(abs(stress+p*np.eye(3)).max())
            task['refined_certificate']=dict(energy=en,forces=forces,stress=stress,fmax=refined_fmax,stress_residual=refined_stress,
                certified=refined_fmax<=plan['fmax'] and (not variable or refined_stress<=plan['stress_tol']))
            if not task['refined_certificate']['certified']:
                task.update(status='refinement_certificate_failed',requests=surface.requests-before);save();continue
            chart2=SymmetricLogStrainChart(b,strain_length=task['strain_length']) if variable else None
            center=chart2.pack(b) if variable else b.positions.ravel().copy();ndof=len(center)
            translations=np.zeros((ndof,3))
            for axis in range(3):translations[axis:3*len(b):3,axis]=1/np.sqrt(len(b))
            u,_,_=np.linalg.svd(translations,full_matrices=True);basis=u[:,3:]
            task.update(status='hessian_partial',spectra=[],hessian_domain='3N+6 minus translations' if variable else '3N minus translations; stress diagnostic only')
            for h in plan['hessian_steps']:
                columns=[];spectrum=dict(h=h,columns=columns,status='pending',dimension=basis.shape[1]);task['spectra'].append(spectrum)
                for direction in basis.T:
                    gradients=[]
                    for sign in (1.,-1.):
                        x=center+sign*h*direction
                        if variable:g=chart2.evaluate(x,surface.evaluate,pressure=p).gradient
                        else:
                            trial=b.copy();trial.positions=x.reshape(-1,3);_,forces,_=surface.evaluate(trial);g=-forces.ravel()
                        gradients.append(g)
                    columns.append(basis.T@(gradients[0]-gradients[1])/(2*h));save()
                matrix=np.asarray(columns).T
                spectrum.update(status='completed',eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2),antisymmetry=float(np.linalg.norm(matrix-matrix.T)))
            task['status']='completed'
        except Exception as exc:task.update(status='pending_or_failed',error=repr(exc))
        task['requests']=surface.requests-before;save()
    report['status']='completed' if endpoints and all(r['source_status']!='missing' for r in runs) and all(t['status']=='completed' for t in report['tasks']) and all(e['fresh_status']=='checked' for e in report['endpoints']) else 'partial'
    save();return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate',required=True);parser.add_argument('--output',required=True);parser.add_argument('--model',required=True);args=parser.parse_args()
    start=time.monotonic()
    runs,endpoints=collect(Path(args.gate)) # No model import before eight terminal processes.
    if any(Path(r['declared_model']).resolve()!=Path(args.model).resolve() for r in runs):
        raise ValueError('qualification model must match all eight declared gate model paths')
    model_hash=hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    if any(r.get('source_model_sha256') not in (None,model_hash) for r in runs):
        raise ValueError('qualification model checksum differs from a gate run')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(dict(PLAN,arguments=vars(args),endpoint_count=len(endpoints),model_sha256=model_hash),indent=2))
    shutil.copy2(__file__,out/'script.py');shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    helpers=out/'source'/'research'/'ga_ssw';helpers.mkdir(parents=True)
    for helper in ('compare_vc_arms.py','analyze_material_gate.py'):
        shutil.copy2(Path(__file__).with_name(helper),helpers/helper)
    from threadpoolctl import threadpool_limits
    thread_guard=threadpool_limits(limits=1)
    import importlib.metadata
    (out/'versions.json').write_text(json.dumps({name:importlib.metadata.version(name) for name in ('mace-torch','torch','ase','numpy','pymatgen')},indent=2))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    calc=MACECalculator(model_paths=args.model,device='cpu',default_dtype='float64')
    class Bounded(ASEStressSurface):
        def evaluate(self,a):
            if self.requests>=PLAN['total_requests'] or time.monotonic()-start>=PLAN['total_seconds']:raise Limit('declared qualification budget exhausted')
            calc.reset() # Every charged request is independent of ASE result cache.
            return super().evaluate(a)
    from research.ga_ssw.analyze_material_gate import matching
    s=Bounded(calc)
    def persist(report):(out/'result.json').write_text(json.dumps(serial(dict(report,wall_seconds=time.monotonic()-start)),indent=2,allow_nan=False))
    qualify(runs,endpoints,s,persist=persist,matching=matching)


if __name__=='__main__':main()
