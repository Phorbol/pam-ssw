"""Prepare a bounded staged-SSW comparison with research Broyden directions.

The public staged presweep and all search/landing code remain unchanged.  The
two research arms replace only ``staged_direction.paper_biased_direction`` in
the main stage.  This file is a protocol generator by default; no PES runs
occur without ``--execute``.
"""
import argparse, hashlib, json, shutil, sys, time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "research/ga_ssw/evidence/verified-ritz-multicase-20260912"

def enc(x):
    from ase import Atoms
    if isinstance(x, Atoms): return dict(numbers=x.numbers.tolist(), positions=x.positions.tolist(), cell=x.cell.array.tolist(), pbc=x.pbc.tolist())
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): enc(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [enc(v) for v in x]
    if hasattr(x, '__dict__'): return enc(vars(x))
    return x

def dump(path, value):
    path.write_text(json.dumps(enc(value), indent=2, allow_nan=False) + '\n')

def append(path, value):
    with path.open('a') as f: f.write(json.dumps(enc(value), allow_nan=False) + '\n')

def atoms_from(payload):
    from ase import Atoms
    return Atoms(numbers=payload['numbers'], positions=payload['positions'],
                cell=payload['cell'], pbc=payload['pbc'])

class CountedSurface:
    """ASE surface with exact request, failure, and denial accounting."""
    def __init__(self, calculator, ledger, cap=3000, wall=90.):
        self.calculator, self.ledger, self.cap, self.wall = calculator, ledger, cap, wall
        self.requests = self.denials = 0; self.started = time.monotonic(); self.boundary = None
    def evaluate(self, atoms):
        if self.requests >= self.cap or time.monotonic()-self.started >= self.wall:
            self.denials += 1; self.boundary = 'request_cap' if self.requests >= self.cap else 'wall_cap'
            append(self.ledger, {'kind':'search_denial','request':self.requests,'reason':self.boundary,'atoms':atoms})
            raise RuntimeError(self.boundary)
        self.requests += 1
        work = atoms.copy(); work.calc = self.calculator
        try:
            energy = float(work.get_potential_energy()); forces = np.asarray(work.get_forces(), float)
            if not np.isfinite(energy) or forces.shape != work.positions.shape or not np.isfinite(forces).all():
                raise ValueError('nonfinite energy/forces')
            append(self.ledger, {'kind':'search','request':self.requests,'energy':energy,
                'fmax':float(np.linalg.norm(forces,axis=1).max()), 'atoms':work,
                'forces':forces})
            return energy, forces
        except Exception as exc:
            append(self.ledger, {'kind':'search_failure','request':self.requests,
                                 'error':repr(exc),'atoms':work})
            raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--execute',action='store_true')
    a=ap.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'source'/'research'/'ga_ssw').mkdir(parents=True)
    for name in ('broyden_direction_reconstruction.py','broyden_state_reconstruction.py','broyden_history_reconstruction.py'):
        shutil.copy2(ROOT/'research/ga_ssw'/name,out/'source'/'research'/'ga_ssw'/name)
    (out/'source'/'research'/'__init__.py').write_text(''); (out/'source'/'research'/'ga_ssw'/'__init__.py').write_text('')
    shutil.copy2(__file__,out/'runner.py')
    sys.path.insert(0,str(out/'source'))
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out/'source')
    from pamssw.standalone import SSWConfig, run_ssw, ASESurface
    import pamssw.standalone.staged_direction as staged
    broyden_module=__import__('research.ga_ssw.broyden_direction_reconstruction',fromlist=['broyden_direction'])
    assert Path(broyden_module.__file__).resolve().is_relative_to(out/'source')
    broyden=broyden_module.broyden_direction
    sources={c:EVIDENCE/f'{c}-verified-seed11/result.json' for c in ('cu13','cu31_fixed','bicyclobutane')}
    if not all(p.exists() for p in sources.values()): raise FileNotFoundError('all verified input results are required')
    inputs={c:json.loads(p.read_text())['initial']['atoms'] for c,p in sources.items()}; dump(out/'inputs.json',inputs)
    dump(out/'input-sources.json',{c:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for c,p in sources.items()})
    base=SSWConfig(width=.1,rotation_bias=None,pre_rotation_hvp=5,max_gaussians=25,temperature_K=150.,fmax=.01,
        bias_fmax=.1,relax_steps=400,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,direction_sampling='global',
        rotation_solver='ritz',cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total',lbfgs_memory=10)
    arms=['ritz','broyden_euclidean','broyden_native']
    plan={'cases':list(inputs),'seeds':[11,29],'arms':arms,'config':asdict(base),'steps':2,'search_cap':3000,'wall_seconds':90,
          'backend':'EMT Cu13/Cu31; GFN2-xTB accuracy .001 bicyclobutane; single CPU thread',
          'intervention':'only staged_direction.paper_biased_direction main-stage callable; public presweep remains plane dimer',
          'broyden_initial_factor':.05,'factor_source':'ELF para.cbdfact static .05 and DESW.quick_setting0; explicit input may override; docs/research/2026-09-12-native-cbdfact-provenance.md','broyden_metrics':{'broyden_euclidean':'euclidean','broyden_native':'native_block_sum'},
          'projected_symmetry_error': '0 means no sampled symmetry matrix in adapter, not an operator-symmetry claim',
          'scope':'complete fixed-cell escapes; developmental comparison, no superiority claim'}
    dump(out/'plan.json',plan)
    dump(out/'source-manifest.json',{'sha256':{str(p.relative_to(out/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')}})
    if not a.execute:return
    from ase.calculators.emt import EMT
    from tblite.ase import TBLite
    original=staged.paper_biased_direction; rows=[]
    for case,payload in inputs.items():
      initial=atoms_from(payload); cfg=replace(base,cluster_frame='translation_only') if initial.pbc.all() else base
      for seed in (11,29):
       for arm in arms:
        folder=out/f'{case}-{arm}-seed{seed}'; folder.mkdir(); ledger=folder/'evaluations.jsonl'; tracefile=folder/'broyden-traces.jsonl'
        def calc(): return TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0) if case=='bicyclobutane' else EMT()
        surface=CountedSurface(calc(),ledger); fresh=None; checks=[]; row={'case':case,'seed':seed,'arm':arm,'status':'started'}
        def adapter(atoms,anchor,**kw):
            metric='euclidean' if arm=='broyden_euclidean' else 'native_block_sum'
            result=broyden(atoms,anchor,rotation_bias=kw['rotation_bias'],fd_step=kw['fd_step'],max_hvp=kw['max_hvp'],
                           tol=kw['tol'],initial_factor=plan['broyden_initial_factor'],metric=metric,evaluate=kw['evaluate'])
            append(tracefile,{'arm':arm,'metric':metric,'result':result})
            return SimpleNamespace(direction=result.direction,curvature=result.curvature,residual_norm=result.residual_norm,
                hvp_calls=result.hvp_calls,force_calls=result.force_calls,converged=result.converged,projected_symmetry_error=0.)
        try:
            if arm!='ritz': staged.paper_biased_direction=adapter
            result=run_ssw(initial.copy(),surface,steps=2,config=cfg,rng=np.random.default_rng(seed))
            assert result.evaluation_requests == surface.requests == result.initial.evaluation_requests + sum(r.evaluation_requests for r in result.records)
            dump(folder/'result.json',result); row.update(status=result.status,search_requests=surface.requests,
                minima=len(result.minima),best_delta=min(q.energy for q in result.minima)-result.initial.energy,record_statuses=dict(Counter(r.status for r in result.records)))
            fresh=ASESurface(calc()); checks=[]
            for i,q in enumerate(result.minima):
                try:
                    fresh.calculator=calc(); e,f=fresh.evaluate(q.atoms); fm=float(np.linalg.norm(f,axis=1).max()); checks.append({'index':i,'energy':e,'error':e-q.energy,'fmax':fm,'qualified':fm<=.01,'cell_unchanged':bool(np.array_equal(q.atoms.cell.array,initial.cell.array))})
                except Exception as exc: checks.append({'index':i,'error':repr(exc)})
            dump(folder/'fresh-checks.json',checks); row['fresh_requests']=fresh.requests; row['fresh_checks']=checks
        except Exception as exc: row.update(status='failed',error=repr(exc))
        finally: staged.paper_biased_direction=original
        row.update(fresh_requests=0 if fresh is None else fresh.requests,fresh_checks=checks,search_requests=surface.requests,denials=surface.denials,boundary=surface.boundary,elapsed=time.monotonic()-surface.started)
        dump(folder/'summary.json',row); rows.append(row); dump(out/'summary.json',rows)
        print(case,seed,arm,row['status'],surface.requests,row.get('minima'),flush=True)

if __name__=='__main__': main()
