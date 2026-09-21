"""Fixed-cell periodic Cu EMT end-to-end implementation check."""
import json
import subprocess
import time
from importlib.metadata import version
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.io import write
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface,SSWConfig,run_ssw
from .compare_dimer_ritz_cu13_escape import serial

def main():
    out=Path('research/ga_ssw/evidence/periodic-cu-ssw');out.mkdir(exist_ok=False)
    (out/'script.py').write_text(Path(__file__).read_text())
    atoms=bulk('Cu','fcc',a=3.6,cubic=True).repeat((2,1,1))
    write(out/'initial.extxyz',atoms)
    cfg=SSWConfig(width=.2,rotation_bias=100,max_gaussians=14,temperature_K=300,fmax=.01,relax_steps=200,
        fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,direction_sampling='global',
        cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    (out/'plan.json').write_text(json.dumps(dict(config=asdict(cfg),seeds=[3,17],steps=5,source='ASE bulk Cu FCC a=3.6 conventional 2x1x1',
        scope='fixed-cell PBC implementation; not VC nor global-search efficiency',backend='ASE EMT'),indent=2))
    sources=out/'source';sources.mkdir()
    for name in ['paper_reference','periodic_geometry','surface','gaussian','dimer']:
        (sources/f'{name}.py').write_text(Path(f'pamssw/standalone/{name}.py').read_text())
    for path in Path('pamssw').rglob('*.py'):
        target=sources/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(path.read_text())
    (out/'git-provenance.json').write_text(json.dumps(dict(
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--porcelain'],text=True),
        recorded_at='before experiments',
        packages={name:version(name) for name in ('ase','numpy','scipy')}),indent=2))
    (out/'working-tree.diff').write_text(subprocess.check_output(['git','diff','HEAD'],text=True))
    rows=[]
    for solver in ['ritz','dimer']:
        for seed in [3,17]:
            from dataclasses import replace
            surface=ASESurface(EMT());config=replace(cfg,rotation_solver=solver)
            search_started=time.monotonic()
            result=run_ssw(atoms,surface,steps=5,config=config,rng=np.random.default_rng(seed))
            search_seconds=time.monotonic()-search_started
            checks=[];fresh=ASESurface(EMT());validation_started=time.monotonic()
            for q in result.minima:
                e,f=fresh.evaluate(q.atoms)
                checks.append(dict(force=float(np.linalg.norm(f,axis=1).max()),energy=e,
                    fixed_cell=bool(np.array_equal(q.atoms.cell,atoms.cell)),pbc=q.atoms.pbc.tolist()))
            failure_statuses=[r.status for r in result.records
                if r.status not in ('gaussian_limit','lower_true_energy')]
            row=dict(seed=seed,solver=solver,requests=surface.requests,
                search_requests=surface.requests,fresh_requests=fresh.requests,
                total_requests=surface.requests+fresh.requests,
                search_seconds=search_seconds,validation_seconds=time.monotonic()-validation_started,
                status=result.status,failed_steps=len(failure_statuses),failure_statuses=failure_statuses,
                force_checks_passed=all(c['force']<=config.fmax for c in checks),
                fixed_cell_checks_passed=all(c['fixed_cell'] and all(c['pbc']) for c in checks),
                steps=[r.status for r in result.records],checks=checks)
            (out/f'{seed}-{solver}.json').write_text(json.dumps(dict(**row,result=serial(result)),indent=2))
            rows.append(row);print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
