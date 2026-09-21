"""Bounded serial CPU VC evidence. Requires explicit backend and input."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from ase.build import bulk
from ase.io import read, write
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface, VCEvaluation
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw


def serial(value):
    if isinstance(value, VCEvaluation):
        return dict(energy=value.energy, objective=value.objective, volume=value.volume,
            fmax=float(np.linalg.norm(value.forces,axis=1).max()),
            stress=value.stress.tolist(), positions=value.atoms.positions.tolist(),
            cell=value.atoms.cell.array.tolist(), symbols=value.atoms.get_chemical_symbols())
    if isinstance(value, np.ndarray): return value.tolist()
    if dataclasses.is_dataclass(value):
        return {f.name:serial(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serial(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['emt','mace'],required=True)
    p.add_argument('--input');p.add_argument('--model')
    p.add_argument('--length',type=float,required=True)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--out',required=True)
    p.add_argument('--request-limit',type=int,default=2000)
    p.add_argument('--seconds',type=float,default=600)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    atoms=read(args.input,index=0) if args.input else bulk('Cu','fcc',a=3.6,cubic=True)
    if args.backend=='emt': calc=EMT()
    else:
        from mace.calculators import MACECalculator
        calc=MACECalculator(model_paths=args.model,device='cpu',default_dtype='float64')
    config=VCSSWConfig(strain_length=args.length,width=.2,rotation_bias=100)
    write(out/'input.extxyz',atoms)
    (out/'provenance.json').write_text(json.dumps(dict(arguments=vars(args),config=dataclasses.asdict(config),
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=subprocess.check_output(['git','status','--short'],text=True),
        model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.model else None),indent=2))
    source=out/'source';source.mkdir()
    for name in ['vc_reference.py','vc_geometry.py','generalized_numerics.py','direction.py']:
        shutil.copy2(Path('pamssw/standalone')/name,source/name)
    shutil.copy2('pamssw/relax.py',source/'relax.py')
    shutil.copytree('pamssw', source/'pamssw', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__,source/'run_joint_vc.py')
    started=time.monotonic()
    class BoundedSurface(ASEStressSurface):
        def evaluate(self, candidate):
            if self.requests >= args.request_limit or time.monotonic()-started >= args.seconds:
                raise RuntimeError('declared CPU preflight budget exhausted')
            values=super().evaluate(candidate)
            if self.requests % 25 == 0:
                (out/'progress.json').write_text(json.dumps(dict(requests=self.requests,
                    seconds=time.monotonic()-started, energy=values[0],
                    fmax=float(np.linalg.norm(values[1],axis=1).max()),
                    volume=candidate.get_volume())))
                write(out/'latest-evaluated.extxyz',candidate)
            return values
    surface=BoundedSurface(calc)
    result=run_vc_ssw(atoms,surface,steps=args.steps,config=config,rng=np.random.default_rng(args.seed))
    elapsed=time.monotonic()-started
    # Fresh oracle calls on every certified minimum; no reuse of cached labels.
    checks=[]
    validation_surface=ASEStressSurface(calc)
    for ev in result.minima:
        calc.reset()
        e,f,s=validation_surface.evaluate(ev.atoms)
        checks.append(dict(energy=e,energy_error=e-ev.energy,
            fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(np.abs(s).max())))
    payload=dict(result=serial(result),search_seconds=elapsed,fresh_checks=checks,
        search_requests=result.requests,fresh_requests=len(checks),total_requests=surface.requests+validation_surface.requests,
        failed_proposals=sum(x.get('status') not in ('gaussian_limit','lower_true_enthalpy')
                             for x in result.records if 'index' in x))
    (out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    if result.minima: write(out/'minima.extxyz',[x.atoms for x in result.minima])
    print(json.dumps({k:v for k,v in payload.items() if k not in ('result','fresh_checks')}))

if __name__=='__main__': main()
