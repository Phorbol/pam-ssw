"""Paired single-step real C60/GFN2-xTB LS-SSW diagnostic, bounded CPU only."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from ase import Atoms
from ase.io import write
from tblite.ase import TBLite
from pamssw.standalone import ASESurface,SSWConfig,LSSettings,run_ls_ssw
from .run_independent_water_ga import serial


def main():
    out=Path('research/ga_ssw/evidence/dimer-ritz-c60-ls')
    out.mkdir(parents=True,exist_ok=False)
    (out/'run_script.py').write_text(Path(__file__).read_text())
    prior=Path('research/ga_ssw/evidence/independent-c60-gfn2-ls')
    atoms=Atoms(**json.loads((prior/'result.json').read_text())['initial']['atoms'])
    old=json.loads((prior/'config.json').read_text())
    ls=LSSettings(bond_energies={(6,6):3.61},bond_lengths={(6,6):old['ls']['bond_lengths']['(6, 6)']} ,target_per_atom=.02)
    base=dict(width=.2,rotation_bias=100.,max_gaussians=2,temperature_K=300.,
              fmax=.01,relax_steps=400,fd_step=.0001,rotation_hvp=100,rotation_tol=.02)
    plan=dict(system='C60',backend='tblite GFN2-xTB',accuracy=.001,seed=20260909,steps=1,
              config=base,ls=serial(ls),solvers=['ritz','dimer'],budget='one CPU; external timeout 300 seconds',
              purpose='paired LS workflow diagnostic, not efficiency or new-basin proof')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');write(out/'initial.extxyz',atoms)
    reports=[]
    for solver in plan['solvers']:
        surface=ASESurface(TBLite(method='GFN2-xTB',verbosity=0,accuracy=.001));started=time.monotonic()
        result=run_ls_ssw(atoms,surface,steps=1,config=SSWConfig(**base,rotation_solver=solver),
                          rng=np.random.default_rng(plan['seed']),ls=ls)
        checks=[]
        for minimum in result.minima:
            e,f=ASESurface(TBLite(method='GFN2-xTB',verbosity=0,accuracy=.001)).evaluate(minimum.atoms)
            checks.append(dict(energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),passed=bool(np.linalg.norm(f,axis=1).max()<=base['fmax'])))
        (out/f'{solver}.json').write_text(json.dumps(dict(result=serial(result),fresh_checks=checks),indent=2)+'\n')
        row=dict(solver=solver,status=result.status,stages=[r.status for r in result.records],requests=surface.requests,
                 fresh_checks=checks,wall_seconds=time.monotonic()-started)
        reports.append(row);print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(runs=reports,distinct_basins_certified=False,physical_stability_certified=False),indent=2)+'\n')

if __name__=='__main__':main()
