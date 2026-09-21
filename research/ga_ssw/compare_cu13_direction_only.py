"""Research-only instantaneous direction projection ablation, not native parity.

Replace direction solvers within this process only. Biased quench, Gaussian,
height and MC remain the Cartesian driver. Each Gaussian rotation uses the
rigid complement at its current center; unlike Eckart runs, no fixed outer
section is imposed on quenching. Scalar settings and seeds remain identical.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from ase.io import read, write
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, run_ssw
from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone import paper_reference, dimer
from .compare_dimer_ritz_cu13_escape import serial


def projected_solver(solver):
    def rotate(atoms,anchor,*,evaluate,**kwargs):
        frame=ClusterFrame(atoms)
        internal=frame.project(anchor);norm=np.linalg.norm(internal)
        if norm<=np.finfo(float).eps*internal.size:
            raise ValueError('anchor has no resolvable internal component')
        internal/=norm
        def projected(candidate):
            candidate=candidate.copy()
            candidate.positions=frame.positions(candidate.positions)
            e,f=evaluate(candidate)
            return e,frame.project(f)
        return solver(atoms,internal,evaluate=projected,**kwargs)
    return rotate


def main():
    out=Path('research/ga_ssw/evidence/cu13-direction-only')
    out.mkdir(exist_ok=False)
    (out/'run_script.py').write_text(Path(__file__).read_text())
    source=out/'source';source.mkdir()
    for name in ['paper_reference.py','cluster_frame.py','surface.py','direction.py','dimer.py','gaussian.py']:
        (source/name).write_text((Path('pamssw/standalone')/name).read_text())
    atoms=read('research/ga_ssw/evidence/independent-cu13-surface/quenched.extxyz')
    write(out/'initial.extxyz',atoms)
    base=dict(width=.2,rotation_bias=100.,max_gaussians=14,temperature_K=300.,
        fmax=.01,relax_steps=200,fd_step=.0001,rotation_hvp=100,rotation_tol=.02)
    plan=dict(system='Cu13',backend='ASE EMT',seeds=[3,17],steps=15,config=base,
        intervention='project anchor and rotation operator at each Gaussian center, unrestricted Cartesian biased quench',
        limits='same outer-step and per-stage limits, NOT matched E/F cost; not exact native setconstraints; changing local versus fixed reference is part of intervention')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    original_ritz=paper_reference.paper_biased_direction;original_dimer=dimer.paper_dimer_direction
    paper_reference.paper_biased_direction=projected_solver(original_ritz)
    dimer.paper_dimer_direction=projected_solver(original_dimer)
    rows=[]
    try:
        for seed in plan['seeds']:
            for solver in ['ritz','dimer']:
                surface=ASESurface(EMT());started=time.monotonic()
                config=SSWConfig(**base,rotation_solver=solver)
                result=run_ssw(atoms,surface,steps=15,config=config,rng=np.random.default_rng(seed))
                checks=[]
                for m in result.minima:
                    e,f=ASESurface(EMT()).evaluate(m.atoms)
                    checks.append(dict(energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),passed=bool(np.linalg.norm(f,axis=1).max()<=base['fmax'])))
                full=dict(seed=seed,solver=solver,intervention=plan['intervention'],config=asdict(config),result=serial(result),fresh_checks=checks)
                (out/f'{seed}-{solver}.json').write_text(json.dumps(full,indent=2)+'\n')
                row=dict(seed=seed,solver=solver,requests=surface.requests,landings=len(result.minima),stages=[r.status for r in result.records],wall_seconds=time.monotonic()-started)
                rows.append(row);print(json.dumps(row),flush=True)
    finally:
        paper_reference.paper_biased_direction=original_ritz
        dimer.paper_dimer_direction=original_dimer
    (out/'summary.json').write_text(json.dumps(dict(runs=rows),indent=2)+'\n')
if __name__=='__main__':main()
