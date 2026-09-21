"""Research four-arm lifecycle runner; no campaign is launched on import.

Each arm performs exactly one matched joint initial quench from the same supplied
raw input. Its preparation cost is charged once to that arm. Block and joint
walkers run as whole chains, preserving their own RNG and proposal schedules.
Caller-owned surfaces enforce budgets; this wrapper adds no retries or PES calls
for diagnostics. CLI uses one CPU thread and explicit request/wall ceilings.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from ase.io import read
from pamssw.standalone.atomic_climb import atomic_climb
from pamssw.standalone.block_ssw import BlockSSWConfig, FixedCellSurface, run_block_ssw
from pamssw.standalone.cell_relax import cell_quench
from pamssw.standalone.generalized_numerics import safe_lbfgs
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw
from research.ga_ssw.compare_vc_arms import serial, mc_accept, BudgetExhausted


class ObservedSurface:
    """Observe caller cost deltas; never count a denied request as a PES call."""
    def __init__(self, surface):
        self.surface=surface
        self.ledger=[]
        self.last=None
        self.censored=False
    @property
    def requests(self):return self.surface.requests
    def evaluate(self, atoms):
        before=self.requests
        try:
            e,f,s=self.surface.evaluate(atoms)
        except Exception as exc:
            self.censored |= isinstance(exc,BudgetExhausted) or bool(getattr(self.surface,'exhausted',False))
            self.ledger.append(dict(before=before,after=self.requests,error=repr(exc)))
            raise
        self.last=(atoms.copy(),float(e),np.array(f).copy(),np.array(s).copy())
        self.ledger.append(dict(before=before,after=self.requests))
        return e,f,s


def validate_configs(atomic, block, joint):
    if block.atomic!=atomic:raise ValueError('block and atomic configs must match')
    for name in ('width','rotation_bias','max_gaussians','temperature_K','forward_force',
                 'fmax','relax_steps','fd_step','rotation_hvp','rotation_tol'):
        if getattr(atomic,name)!=getattr(joint,name):raise ValueError('unmatched '+name)
    for left,right in [('quench_length','strain_length'),('pressure','pressure'),
                       ('stress_tol','stress_tol'),('max_step','max_step')]:
        if getattr(block,left)!=getattr(joint,right):raise ValueError('unmatched '+left)
    if atomic.rotation_solver!='dimer':raise ValueError('joint uses plane dimer; use dimer for atomic comparison')


def run_material_arm(atoms, surface, *, arm, atomic_config, block_config,
                     joint_config, steps, seed, direction_solver=None,
                     rotation_force_calls=None):
    """Return raw stage records and all valid landings, including MC rejects.

    Supply the SAME raw Atoms and matched configs to each independent arm.
    A supplied ASE Calculator is wrapped as ASEStressSurface; a counted EFS
    surface is used unchanged. Budget surfaces should raise BudgetExhausted or
    set exhausted=True when denying calls. Other failures remain failures.
    """
    if arm not in ('fixed','pqc','block','joint'):raise ValueError('unknown arm')
    if isinstance(steps,bool) or not isinstance(steps,int) or steps<0:raise ValueError('nonnegative integer steps required')
    validate_configs(atomic_config,block_config,joint_config)
    if not hasattr(surface,'requests'):surface=ASEStressSurface(surface)
    s=ObservedSurface(surface);begin=s.requests;rng=np.random.default_rng(seed)
    c=atomic_config;b=block_config
    report=dict(arm=arm,seed=seed,steps_requested=steps,input=atoms.copy(),
        objective='E' if arm=='fixed' else 'E+pV',pressure=b.pressure,
        domain='fixed at common prepared cell' if arm=='fixed' else 'variable cell',
        initialization='one matched joint quench from supplied raw input per arm; physically charged once',
        atomic_config=asdict(c),block_config=asdict(b),joint_config=asdict(joint_config),
        records=[],landings=[],current=None,best=None,status='running')
    def joint_quench(a):
        return cell_quench(a,s,strain_length=b.quench_length,pressure=b.pressure,
            fmax=c.fmax,stress_tol=b.stress_tol,max_step=b.max_step,maxiter=c.relax_steps)
    def point(ev,certificate,index,accepted):
        return dict(atoms=ev.atoms.copy(),energy=ev.energy,
            objective=ev.energy if arm=='fixed' else ev.objective,
            forces=ev.forces.copy(),stress=ev.stress.copy(),certificate=certificate,
            index=index,accepted=bool(accepted))
    if arm in ('block','joint'):
        kernel=(run_block_ssw(atoms,s,steps=steps,config=b,rng=rng) if arm=='block'
            else run_vc_ssw(atoms,s,steps=steps,config=joint_config,rng=rng,
                            direction_solver=direction_solver,
                            rotation_force_calls=rotation_force_calls))
        report['kernel']=kernel;report['records']=kernel.records;report['status']=kernel.status
        if kernel.minima:
            initial=kernel.minima[0]
            cert=kernel.records[0]['quench'].certificate if arm=='block' else kernel.records[0]['certificate']
            report['landings'].append(point(initial,cert,-1,True))
            report['common_start']=initial.atoms.copy()
            for event in kernel.records[1:]:
                landing=event['landing']
                if arm=='block':
                    valid=landing is not None and landing.converged
                    ev=landing.evaluation if valid else None
                    cert=landing.certificate if valid else None
                else:
                    cert=event['certificate'];ev=landing
                    valid=(landing is not None and cert is not None and cert['certified']
                        and event['status'] in ('gaussian_limit','lower_true_enthalpy',
                                                'biased_numerical_stop'))
                if valid:report['landings'].append(point(ev,cert,event['index'],event['accepted']))
            report['current']=kernel.current;report['best']=kernel.best
    else:
        initial=joint_quench(atoms)
        report['records'].append(dict(stage='initial',quench=initial,requests=initial.requests))
        if not initial.converged:report['status']='initial_quench_failed'
        else:
            current=point(initial.evaluation,initial.certificate,-1,True);best=current
            report['common_start']=current['atoms'].copy();report['landings'].append(current)
            fixed=FixedCellSurface(s)
            for index in range(steps):
                if s.censored:break
                before=s.requests;record=dict(index=index,accepted=False,status='running',atomic=None,landing=None)
                try:
                    work=current['atoms'].copy()
                    reference=current['objective']-(b.pressure*work.get_volume() if arm=='pqc' else 0.)
                    climb=atomic_climb(work,fixed,reference_energy=reference,config=c,rng=rng)
                    record['atomic']=climb
                    if climb.status not in ('gaussian_limit','lower_true_energy'):
                        record['status']='atomic_'+climb.status
                    elif arm=='pqc':
                        landing=joint_quench(climb.atoms);record['landing']=landing
                        record['status']='valid_landing' if landing.converged else 'true_quench_failed'
                        if landing.converged:candidate=point(landing.evaluation,landing.certificate,index,False)
                    else:
                        start=s.requests;trial=climb.atoms.copy()
                        def evaluate(x):
                            trial.positions=x.reshape(-1,3)
                            e,f=fixed.evaluate(trial);return e,-f.ravel()
                        norm=lambda x:float(np.linalg.norm(x.reshape(-1,3),axis=1).max())
                        relaxed=safe_lbfgs(trial.positions.ravel(),evaluate,gradient_norm=norm,step_norm=norm,
                            gtol=c.fmax,max_step=b.max_step,maxiter=c.relax_steps)
                        trial.positions=relaxed.q.reshape(-1,3)
                        record['landing']=dict(optimizer=relaxed,atoms=trial.copy(),requests=s.requests-start)
                        if relaxed.energy is None:record['status']='true_quench_failed'
                        else:
                            e,f,stress=s.evaluate(trial) # counted in-run certificate request; backend cache may satisfy it
                            certificate=dict(fmax=norm(f.ravel()),stress_max=float(abs(stress+b.pressure*np.eye(3)).max()),
                                stress_required=False,certified=norm(f.ravel())<=c.fmax)
                            record['landing'].update(energy=e,forces=f,stress=stress,certificate=certificate,requests=s.requests-start)
                            record['status']='valid_landing' if relaxed.converged and certificate['certified'] else 'true_quench_failed'
                            candidate=dict(atoms=trial.copy(),energy=e,objective=e,forces=f,stress=stress,
                                certificate=certificate,index=index,accepted=False)
                    if record['status']=='valid_landing':
                        delta=candidate['objective']-current['objective']
                        accepted,draw=mc_accept(delta,c.temperature_K,rng)
                        candidate['accepted']=bool(accepted);report['landings'].append(candidate)
                        if candidate['objective']<best['objective']:best=candidate
                        if accepted:current=candidate
                        record.update(accepted=bool(accepted),delta=delta,mc_draw=draw)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    record.update(status='evaluation_failed',error=repr(exc))
                record['requests']=s.requests-before;report['records'].append(record)
            report.update(current=current,best=best,status='completed')
    report['requests']=s.requests-begin
    report['initial_requests']=report['records'][0]['requests']
    report['ledger']=s.ledger
    report['status']='censored' if s.censored else report['status']
    report['valid_proposals']=max(0,len(report['landings'])-1)
    report['requests_reconciled']=report['requests']==sum(e['requests'] for e in report['records'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--format');parser.add_argument('--config',required=True)
    parser.add_argument('--arm',choices=('fixed','pqc','block','joint'),required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--steps',required=True,type=int)
    parser.add_argument('--cap',required=True,type=int);parser.add_argument('--seconds',required=True,type=float)
    args=parser.parse_args()
    if args.cap<1 or not np.isfinite(args.seconds) or args.seconds<=0:parser.error('positive cap/seconds required')
    cfg=json.loads(Path(args.config).read_text());atomic=SSWConfig(**cfg['atomic'])
    block=BlockSSWConfig(atomic=atomic,**cfg['block']);joint=VCSSWConfig(**cfg['joint'])
    validate_configs(atomic,block,joint)
    atoms=read(args.input,index=0,format=args.format)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=dict(arguments=vars(args),config=cfg,model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),input=serial(atoms),status='prepared')
    (out/'plan.json').write_text(json.dumps(manifest,indent=2))
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'script.py')
    research_source=out/'source'/'research'/'ga_ssw';research_source.mkdir(parents=True)
    shutil.copy2(__file__,research_source/'compare_material_arms.py')
    shutil.copy2(Path(__file__).with_name('compare_vc_arms.py'),research_source/'compare_vc_arms.py')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    calc=MACECalculator(model_paths=args.model,device='cpu',default_dtype='float64')
    start=time.monotonic()
    class BoundedSurface(ASEStressSurface):
        exhausted=False
        def evaluate(self,a):
            if self.requests>=args.cap or time.monotonic()-start>=args.seconds:
                self.exhausted=True;raise BudgetExhausted('declared material comparison budget exhausted')
            return super().evaluate(a)
    surface=BoundedSurface(calc)
    result=run_material_arm(atoms,surface,arm=args.arm,atomic_config=atomic,block_config=block,
        joint_config=joint,steps=args.steps,seed=args.seed)
    result['wall_seconds']=time.monotonic()-start
    (out/'result.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))


if __name__=='__main__':main()
