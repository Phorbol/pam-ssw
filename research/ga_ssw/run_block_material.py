"""Bounded complex-material block-SSW diagnostic; explicit prototype settings."""
import argparse,dataclasses,json,os,shutil,time
from pathlib import Path
import numpy as np
from ase.io import read,write
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.block_ssw import BlockSSWConfig,run_block_ssw
from pamssw.standalone.paper_reference import SSWConfig
from research.ga_ssw.run_joint_vc import serial as vc_serial
from research.ga_ssw.compare_vc_arms import serial


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--model',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=3)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--cap',type=int,default=1500)
    p.add_argument('--seconds',type=float,default=300)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    atomic=SSWConfig(width=.6,rotation_bias=100,max_gaussians=10,temperature_K=300,
        fmax=.01,relax_steps=300,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,
        direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',
        quench_optimizer='safe-lbfgs-total')
    config=BlockSSWConfig(atomic=atomic,quench_length=5.)
    (out/'plan.json').write_text(json.dumps(dict(arguments=vars(args),config=dataclasses.asdict(config),
        status='bounded_development_not_performance_evaluation'),indent=2))
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'script.py')
    atoms=read(args.input,index=0);write(out/'input.extxyz',atoms)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    calc=MACECalculator(model_paths=args.model,device='cpu',default_dtype='float64')
    start=time.monotonic()
    class Surface(ASEStressSurface):
        def evaluate(self,a):
            if self.requests>=args.cap or time.monotonic()-start>=args.seconds:
                raise RuntimeError('declared development request/wall budget exhausted')
            value=super().evaluate(a)
            if self.requests%25==0:
                (out/'progress.json').write_text(json.dumps(dict(requests=self.requests,
                    seconds=time.monotonic()-start,energy=value[0],fmax=float(np.linalg.norm(value[1],axis=1).max()),volume=a.get_volume())))
                write(out/'last-evaluated.extxyz',a)
            return value
    surface=Surface(calc)
    try:
        result=run_block_ssw(atoms,surface,steps=args.steps,config=config,rng=np.random.default_rng(args.seed))
        seconds=time.monotonic()-start
        checks=[];fresh=ASEStressSurface(calc)
        for ev in result.minima:
            calc.reset();e,f,s=fresh.evaluate(ev.atoms)
            checks.append(dict(energy_error=e-ev.energy,fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(np.abs(s).max())))
        payload=dict(result=serial(result),fresh_checks=checks,search_requests=surface.requests,
            fresh_requests=fresh.requests,search_seconds=seconds)
        if result.minima:write(out/'minima.extxyz',[ev.atoms for ev in result.minima])
    except Exception as exc:
        payload=dict(error=repr(exc),search_requests=surface.requests,search_seconds=time.monotonic()-start)
    (out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in payload.items() if k not in ('result','fresh_checks')}))

if __name__=='__main__':main()
