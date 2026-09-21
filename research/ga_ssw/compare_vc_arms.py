"""Research-only fixed/PQC/joint VC wiring, Cu EMT; no performance claim."""
import argparse,json,math,time
from dataclasses import asdict,is_dataclass
from pathlib import Path
import numpy as np
from ase import Atoms,units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig,run_vc_ssw
from pamssw.standalone.paper_reference import SSWConfig,run_ssw

class BudgetExhausted(RuntimeError):pass

def serial(value):
    if isinstance(value,Atoms):return dict(numbers=value.numbers.tolist(),positions=value.positions.tolist(),cell=value.cell.array.tolist(),pbc=value.pbc.tolist())
    if is_dataclass(value):return {k:serial(v) for k,v in vars(value).items()}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value

class CappedEMT(ASEStressSurface):
    def __init__(self,cap,deadline=None):
        super().__init__(EMT());self.deadline=deadline;self.cap=cap;self.exhausted=False;self.ledger=[];self.cache={};self.stage='setup'
    @staticmethod
    def key(a):return a.numbers.tobytes()+a.positions.tobytes()+a.cell.array.tobytes()+a.pbc.tobytes()
    def evaluate(self,atoms):
        if self.deadline is not None and time.monotonic()>=self.deadline:
            self.exhausted=True;raise BudgetExhausted('global wall deadline reached')
        if self.requests>=self.cap:
            self.exhausted=True;raise BudgetExhausted('arm E/F/stress request cap reached')
        before=self.requests
        try:
            e,f,s=super().evaluate(atoms)
            item=dict(request=self.requests,stage=self.stage,energy=e,forces=f,stress=s,atoms=atoms.copy())
            self.cache[self.key(atoms)]=item;self.ledger.append(item);return e,f,s
        except Exception as exc:
            # ASEStressSurface increments immediately before the calculator;
            # pre-validation failure is a rejected input, not a backend call.
            self.ledger.append(dict(request=self.requests,stage=self.stage,error=repr(exc),charged=self.requests>before));raise

class FixedSurface:
    def __init__(self,parent):self.parent=parent
    @property
    def requests(self):return self.parent.requests
    def evaluate(self,atoms):
        e,f,_=self.parent.evaluate(atoms);return e,f

def mc_accept(delta,temperature,rng):
    if delta<=0:return True,None
    if temperature==0:return False,None
    draw=float(rng.random());return draw<math.exp(-delta/(units.kB*temperature)),draw

def run_arm(atoms,*,arm,fixed_config,vc_config,steps,request_cap,seed,output=None,deadline=None):
    if arm not in ('fixed','posterior-cell-quench','joint-vc'):raise ValueError('unknown arm')
    if set(atoms.get_chemical_symbols())!={'Cu'}:raise ValueError('prototype supports Cu EMT only')
    if steps<0 or request_cap<1:raise ValueError('nonnegative steps and positive request cap required')
    if fixed_config.temperature_K!=vc_config.temperature_K:raise ValueError('all arms require common MC temperature')
    if fixed_config.fmax!=vc_config.fmax:raise ValueError('all arms require common physical force threshold')
    surface=CappedEMT(request_cap,deadline);proposal_seed,mc_seed=np.random.SeedSequence(seed).spawn(2)
    proposal_seeds=proposal_seed.spawn(steps+1);mc_rng=np.random.default_rng(mc_seed)
    report=dict(arm=arm,status='running',seed=seed,request_cap=request_cap,steps_requested=steps,
        domain='fixed cell at common relaxed input' if arm=='fixed' else 'variable cell',
        objective='E' if arm=='fixed' else 'E+pV',pressure=vc_config.pressure,
        fixed_config=asdict(fixed_config),vc_config=asdict(vc_config),input=atoms.copy(),stages=[],landings=[],records=[])
    def persist():
        report['requests']=surface.requests;report['ledger']=surface.ledger
        if output is not None:Path(output).write_text(json.dumps(serial(report),indent=2)+'\n')
    def stage(name,operation):
        surface.stage=name;before=surface.requests
        try:return operation()
        finally:
            report['stages'].append(dict(name=name,requests=surface.requests-before));persist()
    def certificate(a,require_stress):
        entry=surface.cache.get(surface.key(a))
        if entry is None:raise RuntimeError('missing exact cached E/F/stress for returned landing')
        fmax=float(np.linalg.norm(entry['forces'],axis=1).max());stress=float(abs(entry['stress']+vc_config.pressure*np.eye(3)).max())
        return dict(fmax=fmax,stress_max=stress,force_pass=fmax<=vc_config.fmax,
                    stress_required=require_stress,stress_pass=stress<=vc_config.stress_tol,
                    certified=fmax<=vc_config.fmax and (not require_stress or stress<=vc_config.stress_tol),evaluation_request=entry['request'])
    def objective(a):
        entry=surface.cache[surface.key(a)];return entry['energy']+(vc_config.pressure*a.get_volume() if arm!='fixed' else 0.)
    try:
        initial=stage('common_joint_initial_quench',lambda:run_vc_ssw(atoms,surface,steps=0,config=vc_config,rng=np.random.default_rng(proposal_seeds[0])))
        report['initial_kernel']=initial
        if initial.status!='completed' or not initial.minima:
            report['status']='censored' if surface.exhausted else 'initial_quench_failed';persist();return report
        current=initial.initial.atoms.copy();current_objective=objective(current);report['common_start']=current.copy();report['current']=current.copy();report['current_objective']=current_objective
        report['landings'].append(dict(step=-1,atoms=current.copy(),objective=current_objective,certificate=certificate(current,True),accepted=True))
        for step in range(steps):
            if surface.requests>=request_cap:surface.exhausted=True;break
            before=surface.requests;record=dict(step=step,status='running',accepted=False)
            report['records'].append(record);candidate=None
            try:
                rng=np.random.default_rng(proposal_seeds[step+1])
                if arm=='joint-vc':
                    kernel=stage(f'{step}:joint_proposal',lambda:run_vc_ssw(current,surface,steps=1,config=vc_config,rng=rng))
                    record['kernel']=kernel
                    if kernel.status=='completed' and len(kernel.records)>1:
                        event=kernel.records[-1];landing=event.get('landing')
                        if landing is not None and event.get('certificate',{}).get('certified',False) and event.get('landing_optimizer',{}).get('status')=='converged':candidate=landing.atoms.copy()
                else:
                    kernel=stage(f'{step}:fixed_proposal',lambda:run_ssw(current,FixedSurface(surface),steps=1,config=fixed_config,rng=rng))
                    record['kernel']=kernel
                    if kernel.records:
                        landing=kernel.records[-1].landing
                        if landing is not None and landing.converged:candidate=landing.atoms.copy()
                    if candidate is not None and arm=='posterior-cell-quench':
                        record['fixed_landing_before_cell_quench']=candidate.copy()
                        post=stage(f'{step}:posterior_cell_quench',lambda:run_vc_ssw(candidate,surface,steps=0,config=vc_config,rng=rng))
                        record['posterior_kernel']=post
                        candidate=post.initial.atoms.copy() if post.status=='completed' and post.minima else None
                if candidate is None:
                    record['status']='censored' if surface.exhausted else 'no_valid_landing'
                else:
                    cert=certificate(candidate,arm!='fixed')
                    if not cert['certified']:raise RuntimeError('kernel valid landing failed physical certificate')
                    energy=objective(candidate);delta=energy-current_objective
                    accept,draw=mc_accept(delta,vc_config.temperature_K,mc_rng)
                    report['landings'].append(dict(step=step,atoms=candidate.copy(),objective=energy,certificate=cert,accepted=bool(accept)))
                    record.update(status='valid_landing',accepted=bool(accept),delta=delta,mc_draw=draw)
                    if accept:current=candidate.copy();current_objective=energy
            except Exception as exc:
                record.update(status='censored' if surface.exhausted else 'failed',error=repr(exc))
            record['requests']=surface.requests-before;report['current']=current.copy();report['current_objective']=current_objective;persist()
            if surface.exhausted:break
        report['status']='censored' if surface.exhausted else 'completed'
    except Exception as exc:report.update(status='censored' if surface.exhausted else 'failed',error=repr(exc))
    persist();return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=2);p.add_argument('--cap',type=int,default=300);p.add_argument('--seed',type=int,default=7);p.add_argument('--pressure',type=float,default=0.);p.add_argument('--deadline',type=float);args=p.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=False,parents=True);(out/'script.py').write_text(Path(__file__).read_text())
    a=read(args.input,index=0) if args.input else bulk('Cu','fcc',a=3.65,cubic=True)
    # Explicit wiring-test preset; no optimal-parameter or matched-domain claim.
    fixed=SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=300.,fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    vc=VCSSWConfig(strain_length=3.6,width=.2,rotation_bias=.5,max_gaussians=1,relax_steps=150,rotation_hvp=41,pressure=args.pressure)
    results=[]
    for arm in ['fixed','posterior-cell-quench','joint-vc']:
        r=run_arm(a,arm=arm,fixed_config=fixed,vc_config=vc,steps=args.steps,request_cap=args.cap,seed=args.seed,output=out/f'{arm}.json',deadline=args.deadline);results.append(dict(arm=arm,status=r['status'],requests=r['requests'],valid_landings=max(0,len(r['landings'])-1)));print(results[-1],flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
