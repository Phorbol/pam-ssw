"""Full Cu13 SSW comparison after frozen-subproblem optimizer selection.

Same initial structure, seeds and scalar limits as direction-only baseline.
This is a trajectory test, not an equal-cost or held-out generalization claim.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from ase.io import read, write
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, run_ssw
from .compare_dimer_ritz_cu13_escape import serial


def main():
    out=Path('research/ga_ssw/evidence/cu13-safe-total')
    out.mkdir(exist_ok=False)
    (out/'run_script.py').write_text(Path(__file__).read_text())
    source=out/'source';source.mkdir()
    for name in ['paper_reference.py','cluster_frame.py','surface.py','direction.py','dimer.py','gaussian.py']:
        (source/name).write_text((Path('pamssw/standalone')/name).read_text())
    (source/'pam-relax.py').write_text(Path('pamssw/relax.py').read_text())
    atoms=read('research/ga_ssw/evidence/independent-cu13-surface/quenched.extxyz')
    write(out/'initial.extxyz',atoms)
    base=dict(width=.2,rotation_bias=100.,max_gaussians=14,temperature_K=300.,
        fmax=.01,relax_steps=200,fd_step=.0001,rotation_hvp=100,rotation_tol=.02,
        cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total')
    plan=dict(system='Cu13',backend='ASE EMT',seeds=[3,17],steps=15,config=base,
        baseline='research/ga_ssw/evidence/cu13-direction-only',
        intervention='existing Safe-total numerical kernel for all quenches; independent SSW walker',
        success_metrics=['failed stages','fresh true force certificates','strict fingerprint groups and internal Hessians','accepted interbasin moves','all physical requests'],
        limits='same scalar and step limits, not matched E/F; this system was used for optimizer selection; no cross-system generalization claim')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    for seed in plan['seeds']:
        for solver in ['ritz','dimer']:
            surface=ASESurface(EMT());started=time.monotonic()
            config=SSWConfig(**base,rotation_solver=solver)
            result=run_ssw(atoms,surface,steps=15,config=config,rng=np.random.default_rng(seed))
            checks=[]
            for m in result.minima:
                e,f=ASESurface(EMT()).evaluate(m.atoms)
                checks.append(dict(energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),passed=bool(np.linalg.norm(f,axis=1).max()<=base['fmax'])))
            full=dict(seed=seed,solver=solver,config=asdict(config),result=serial(result),fresh_checks=checks)
            (out/f'{seed}-{solver}.json').write_text(json.dumps(full,indent=2)+'\n')
            row=dict(seed=seed,solver=solver,requests=surface.requests,validation_requests=len(checks),landings=len(result.minima),stages=[r.status for r in result.records],accepted=sum(r.accepted for r in result.records),fresh_passed=sum(c['passed'] for c in checks),wall_seconds=time.monotonic()-started)
            rows.append(row);print(json.dumps(row),flush=True)
            (out/'summary.json').write_text(json.dumps(dict(runs=rows,complete=len(rows)==4),indent=2)+'\n')
if __name__=='__main__':main()
