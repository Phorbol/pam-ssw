"""Prepare an ASE BasinHopping baseline; PES execution requires ``--execute``.

The wrapper preserves ASE's move and Metropolis rules, replacing only its
local optimizer with PAM's existing Safe-total quench so local E/F accounting
uses the same backend family as the SSW comparison.
"""
import argparse, hashlib, json, shutil, sys, time, inspect
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--source-root', type=Path,
        default=Path('research/ga_ssw/evidence/two-stage-ritz-comparison-20260912'))
    args = ap.parse_args(); out=args.output.resolve(); src=args.source_root.resolve()
    out.mkdir(parents=False, exist_ok=False)
    frozen_pamssw = src / 'source' / 'pamssw'
    if not frozen_pamssw.is_dir():
        raise FileNotFoundError(f"frozen PAM source missing: {frozen_pamssw}")
    shutil.copytree(frozen_pamssw, out/'source'/'pamssw')
    shutil.copy2(Path(__file__),out/'runner.py'); sys.path.insert(0,str(out/'source'))
    from ase import __version__ as ase_version, units
    from ase.io import read, write
    from ase.calculators.emt import EMT
    from ase.optimize.basin import BasinHopping
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out / "source")
    from pamssw.standalone.surface import ASESurface, quench, SurfaceCalculator
    from research.ga_ssw.compare_vc_arms import serial
    from tblite.ase import TBLite

    source_inputs={}
    for name in ('cu13','cu31_fixed','bicyclobutane'):
        p=src/f'{name}.extxyz'
        if not p.exists(): raise FileNotFoundError(p)
        source_inputs[name]=read(p)
    config=dict(temperature_K=150., dr_A=0.5, fmax=.01, local_steps=400,
        search_cap=6000, wall_seconds=90, seeds=[11,29],
        cases={'cu13':'EMT','cu31_fixed':'EMT','bicyclobutane':'GFN2-xTB'},
        source_inputs={k:str(src/f'{k}.extxyz') for k in source_inputs},
        purpose='ASE BasinHopping baseline; no production/default change',
        note='ASE BasinHopping uses global NumPy RandomState and computes prequench ro in its framework; this is recorded explicitly',
        optimizer='PAM Safe-total adapter; any local failure terminates that arm with paid ledger retained',
        displacement_source='ASE official optimize documentation BasinHopping example dr=.5 A; not fitted; https://ase.gitlab.io/ase/ase/optimize.html',
        random_generator='NumPy global RandomState seeded per arm; not identical random streams to SSW default_rng',
        outer_steps=400)
    (out/'plan.json').write_text(json.dumps(config,indent=2)+'\n')
    for name,a in source_inputs.items(): write(out/f'{name}.extxyz',a)
    from ase.optimize.basin import BasinHopping
    basin_source = Path(inspect.getsourcefile(BasinHopping)).resolve()
    (out/'ase-source'/'ase'/'optimize').mkdir(parents=True)
    shutil.copy2(basin_source, out/'ase-source'/'ase'/'optimize'/'basin.py')
    (out/'source-manifest.json').write_text(json.dumps({'ase_version':ase_version,
        'ase_basin_source':str(basin_source), 'ase_basin_sha256':hashlib.sha256(basin_source.read_bytes()).hexdigest(),
        'pamssw_sha256':{str(p.relative_to(out/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (out/'source').rglob('*.py')}},indent=2)+'\n')
    if not args.execute: return

    class SafeTotalOptimizer:
        def __init__(self, optimizable, *, surface, fmax, steps, results_path, results):
            self.optimizable=optimizable; self.surface=surface; self.fmax=fmax; self.steps=steps
            self.results_path=results_path; self.results=results
            self.result=None
        def __enter__(self): return self
        def __exit__(self,*unused): return False
        def run(self, fmax=None, steps=None):
            before = self.surface.requests
            self.result=quench(self.optimizable.atoms, self.surface,
                fmax=self.fmax if fmax is None else fmax,
                steps=self.steps if steps is None else steps,
                optimizer='safe-lbfgs-total')
            assert self.result.evaluation_requests == self.surface.requests - before
            self.results.append(self.result)
            self.results_path.write_text(json.dumps(serial(self.results), indent=2) + '\n')
            if not self.result.converged: raise RuntimeError('failed local Safe-total quench')
            self.optimizable.atoms.set_positions(self.result.atoms.positions)
            return True

    for name, initial in source_inputs.items():
        for seed in (11,29):
            folder=out/f'{name}-seed{seed}'; folder.mkdir(); started=time.monotonic(); ledger=folder/'evaluations.jsonl'
            calc=lambda: TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0) if name=='bicyclobutane' else EMT()
            local_results=[]; local_results_path=folder/'local-results.json'
            class Counted(ASESurface):
                denied=0; boundary=None
                def evaluate(self,atoms):
                    if self.requests>=6000 or time.monotonic()-started>=90:
                        self.denied+=1; self.boundary='request_cap' if self.requests>=6000 else 'wall_cap';
                        with ledger.open('a') as h: h.write(json.dumps(serial(dict(kind='search_denial',request=self.requests,error=self.boundary,atoms=atoms)))+'\n')
                        raise RuntimeError(self.boundary)
                    try: e,f=super().evaluate(atoms); row=dict(kind='search',request=self.requests,energy=e,forces=f,atoms=atoms)
                    except Exception as error:
                        row=dict(kind='search_failure',request=self.requests,error=repr(error),atoms=atoms)
                        with ledger.open('a') as h: h.write(json.dumps(serial(row))+'\n')
                        raise
                    with ledger.open('a') as h: h.write(json.dumps(serial(row))+'\n')
                    return e,f
            surface=Counted(calc()); np.random.seed(seed)
            def factory(opt, logfile=None):
                return SafeTotalOptimizer(opt,surface=surface,fmax=.01,steps=400,
                                          results_path=local_results_path,results=local_results)
            row={'case':name,'seed':seed,'ase_version':ase_version,'numpy_seed':seed}
            error_text=None
            try:
                working=initial.copy(); working.calc=SurfaceCalculator(surface)
                bh=BasinHopping(working,temperature=150.*units.kB,dr=.5,fmax=.01,
                    logfile=None,trajectory=None,optimizer=factory,optimizer_logfile=None,
                    local_minima_trajectory=None)
                bh.run(400)
                status='completed'
            except Exception as error:
                error_text=repr(error); status='exception'
            fresh=None; checks=[]
            try:
                # Every converged local landing (including MC rejects) is fresh-checked.
                fresh=ASESurface(calc()); checks=[]
                for i,item in enumerate(local_results):
                    if not item.converged: continue
                    try:
                        atoms=item.atoms.copy(); fresh.calculator=calc(); e,f=fresh.evaluate(atoms)
                        checks.append(dict(index=i,energy=e,fmax=float(np.linalg.norm(f,axis=1).max()),
                                           energy_error=e-item.energy,force_qualified=bool(np.linalg.norm(f,axis=1).max() <= .01)))
                    except Exception as error: checks.append(dict(index=i,error=repr(error)))
                    (folder/'fresh-checks.json').write_text(json.dumps(serial(checks),indent=2)+'\n')
                row.update(status=status,search_requests=surface.requests,local_quenches=len(local_results),
                    local_failures=sum(not x.converged for x in local_results),
                    best_energy=min((x.energy for x in local_results if x.converged),default=None),
                    fresh_requests=fresh.requests,fresh_checks=checks)
            except Exception as error: row.update(status='exception',error=repr(error),search_requests=surface.requests,
                fresh_requests=fresh.requests if fresh is not None else 0,fresh_checks=checks)
            if error_text is not None: row['controller_error']=error_text
            (folder/'result.json').write_text(json.dumps(serial({'local_results':local_results,'controller_error':error_text}),indent=2)+'\n')
            if surface.boundary is not None: row['status']='budget_cap'
            ledger_rows=[json.loads(line) for line in ledger.open()] if ledger.exists() else []
            paid=[x for x in ledger_rows if x['kind'] in ('search','search_failure')]
            row.update(ledger_search_rows=len(paid),
                ledger_sequential=[x['request'] for x in paid]==list(range(1,surface.requests+1)))
            row.update(denied=surface.denied,boundary=surface.boundary,ledger_count=len(ledger_rows),
                       wall_seconds=time.monotonic()-started)
            (folder/'summary.json').write_text(json.dumps(serial(row),indent=2)+'\n')


if __name__=='__main__': main()
