"""Paired real Cu13/EMT SSW workflow, explicit budgets, not efficacy campaign."""
from dataclasses import asdict, fields, is_dataclass
import json
from pathlib import Path
import time
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read, write
from pamssw.standalone import ASESurface, SSWConfig, run_ssw
def serial(value):
    if isinstance(value, Atoms):
        return dict(numbers=value.numbers.tolist(),positions=value.positions.tolist(),
                    cell=value.cell.tolist(),pbc=value.pbc.tolist())
    if is_dataclass(value):
        return {f.name:serial(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def main():
    out=Path('research/ga_ssw/evidence/dimer-ritz-cu13')
    out.mkdir(parents=True,exist_ok=False)
    (out/'run_script.py').write_text(Path(__file__).read_text())
    atoms=read('research/ga_ssw/evidence/independent-cu13-surface/quenched.extxyz')
    write(out/'initial.extxyz',atoms)
    base=dict(width=.2,rotation_bias=100.,max_gaussians=2,temperature_K=300.,
              fmax=.01,relax_steps=200,fd_step=.0001,rotation_hvp=100,rotation_tol=.02)
    plan=dict(system='Cu13',backend='ASE EMT',seeds=[3,17,20260909],steps=2,config=base,
              solvers=['ritz','dimer'],acceptance='same paper MC',
              purpose='paired workflow/force-cost diagnostic, not statistically validated efficiency',
              note='same initial structure and RNG seed; later trajectories may diverge; no parameter retuning')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    reports=[]
    for seed in plan['seeds']:
        for solver in plan['solvers']:
            surface=ASESurface(EMT());started=time.monotonic()
            config=SSWConfig(**base,rotation_solver=solver)
            result=run_ssw(atoms,surface,steps=plan['steps'],config=config,rng=np.random.default_rng(seed))
            checks=[]
            for minimum in result.minima:
                e,f=ASESurface(EMT()).evaluate(minimum.atoms)
                checks.append(dict(energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),passed=bool(np.linalg.norm(f,axis=1).max()<=base['fmax'])))
            full=dict(seed=seed,solver=solver,config=asdict(config),result=serial(result),fresh_checks=checks)
            (out/f'{seed}-{solver}.json').write_text(json.dumps(full,indent=2)+'\n')
            row=dict(seed=seed,solver=solver,status=result.status,stages=[r.status for r in result.records],
                     requests=surface.requests,landings=len(result.minima),fresh_checks=checks,wall_seconds=time.monotonic()-started)
            reports.append(row);print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(runs=reports,distinct_basins_certified=False,physical_stability_certified=False),indent=2)+'\n')

if __name__=='__main__':main()
