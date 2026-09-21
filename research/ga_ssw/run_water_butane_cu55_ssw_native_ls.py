"""Bounded fixed-cell SSW/native-LS comparison; preparation unless --execute."""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, shutil, signal, sys, time
from pathlib import Path

import numpy as np
from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.collections import g2
from ase.io import read, write

WATER_ARC = Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE3-(H2O)15/addition/add.arc')
OUT_DEFAULT = Path('research/ga_ssw/evidence/water-bicyclobutane-cu55-ssw-native-ls-20260912')
CU_E, CU_L = 3.6298000812530518, 1.875


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)): return value.item()
    if dataclasses.is_dataclass(value): return {f.name: serial(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, 'get_positions') and hasattr(value, 'get_atomic_numbers'):
        return dict(numbers=value.get_atomic_numbers().tolist(), positions=value.get_positions().tolist(),
                    cell=value.cell.array.tolist(), pbc=value.pbc.tolist(),
                    constraints=[serial(c.todict()) for c in value.constraints])
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
    return value


def dump(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, default=OUT_DEFAULT); ap.add_argument('--execute', action='store_true')
    ap.add_argument('--cases', nargs='+', choices=('water15', 'bicyclobutane', 'cu55'))
    ap.add_argument('--total-wall-seconds', type=float, default=240.)
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(parents=False, exist_ok=False)
    water = read(WATER_ARC, index=0); water.set_pbc(False); water.set_cell(np.zeros((3, 3)))
    cases = {'water15': water, 'bicyclobutane': g2['bicyclobutane'].copy(), 'cu55': Octahedron('Cu', 5, cutoff=2)}
    if args.cases is not None: cases = {name: cases[name] for name in args.cases}
    plan = dict(status='prepared' if not args.execute else 'executing', cases=list(cases), variants=['ssw','native_ls'], seeds=[11,29],
        steps=20, search_request_cap=6000, fresh_cap=21, arm_wall_seconds=60, total_wall_seconds=args.total_wall_seconds,
        backend={'molecules':'tblite GFN2-xTB accuracy .001', 'cu55':'ASE EMT'}, threads=1,
        config_by_case={'water15':dict(width=.1,max_gaussians=25,temperature_K=50.),
                        'bicyclobutane':dict(width=.1,max_gaussians=25,temperature_K=150.),
                        'cu55':dict(width=.2,max_gaussians=14,temperature_K=200.)},
        shared={'rotation_bias':100.,'rotation_solver':'dimer','direction_sampling':'global','cluster_frame':'direction_only',
                'fd_step':1e-4,'rotation_tol':.02,'rotation_hvp':100,'quench_optimizer':'safe-lbfgs-total',
                'lbfgs_memory':None,'fmax':.03,'bias_fmax':.1,'relax_steps':300,'LS_prequench':dict(fmax=.1,steps=300)},
        ls={'water15':dict(table='HCO recovered',target_mev_per_atom=20.),
            'bicyclobutane':dict(table='HCO recovered',target_mev_per_atom=700.),
            'cu55':dict(CuCu_energy=CU_E,CuCu_length=CU_L,target_mev_per_atom=20.,source='native-ls-pair-table-extended')},
        scope='bounded diagnostic, no retries/tuning; failures and censored arms retained; no effectiveness claim')
    dump(out/'plan.json', plan)
    for name, atoms in cases.items(): write(out/f'{name}-input.extxyz', atoms)
    if not args.execute: return
    source = out/'source'/'pamssw'; shutil.copytree('pamssw', source, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__), out/'runner.py')
    dump(out/'initial.json', cases)
    dump(out/'source-manifest.json', {'source':str(source), 'hashes':{str(p.relative_to(out/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')}})
    sys.path.insert(0, str(out/'source')); import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(source.parent)
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw, NativeLSSettings, run_native_ls_ssw
    from pamssw.standalone.native_ls import HCO_BOND_ENERGIES, HCO_BOND_LENGTHS
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from tblite.ase import TBLite
    started_all=time.monotonic(); summaries=[]
    for name, initial in cases.items():
      for variant in ('ssw','native_ls'):
       for seed in (11,29):
        folder=out/f'{name}-{variant}-seed{seed}'; folder.mkdir(); ledger=folder/'evaluations.jsonl'; started=time.monotonic(); result=None; denied=0
        molecular=name!='cu55'; calc=lambda: TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0) if molecular else EMT()
        class Counted(ASESurface):
            def evaluate(self, atoms):
                nonlocal denied
                if self.requests>=6000 or time.monotonic()-started>=60 or time.monotonic()-started_all>=args.total_wall_seconds:
                    denied+=1; ledger.open('a').write(json.dumps({'kind':'denial','requests':self.requests})+'\n'); raise RuntimeError('bounded cap')
                try: e,f=super().evaluate(atoms)
                except Exception as exc:
                    ledger.open('a').write(json.dumps({'kind':'failure','request':self.requests,'error':repr(exc),'atoms':serial(atoms)})+'\n'); raise
                with ledger.open('a') as h: h.write(json.dumps({'kind':'paid','request':self.requests,'energy':e,'forces':f.tolist(),'atoms':serial(atoms)})+'\n')
                return e,f
        row=dict(case=name,variant=variant,seed=seed,status='not_run')
        if time.monotonic()-started_all>=args.total_wall_seconds: row['status']='not_run_global_wall'; dump(folder/'summary.json',row); summaries.append(row); continue
        surface=Counted(calc())
        def alarm_handler(signum, frame):
            raise TimeoutError('per-arm wall cap')
        old_alarm_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, min(60., args.total_wall_seconds - (time.monotonic()-started_all))))
        fresh=[]
        try:
            p=plan['config_by_case'][name]; cfg=SSWConfig(width=p['width'],rotation_bias=100.,max_gaussians=p['max_gaussians'],temperature_K=p['temperature_K'],fmax=.03,bias_fmax=.1,relax_steps=300,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,rotation_solver='dimer',cluster_frame='direction_only',direction_sampling='global',quench_optimizer='safe-lbfgs-total')
            checkpoint_path = folder / 'checkpoint.pkl'
            if variant=='ssw': result=run_ssw(initial.copy(),surface,steps=20,config=cfg,rng=np.random.default_rng(seed),checkpoint_path=checkpoint_path)
            else:
                if name=='cu55': e,l={(29,29):CU_E},{(29,29):CU_L}
                else: e,l=HCO_BOND_ENERGIES,HCO_BOND_LENGTHS
                target=20. if name=='water15' else (700. if name=='bicyclobutane' else 20.)
                ls=NativeLSSettings(e,l,target_mev_per_atom=target,prequench=LSPrequenchSettings(fmax=.1,steps=300))
                result=run_native_ls_ssw(initial.copy(),surface,steps=20,config=cfg,rng=np.random.default_rng(seed),ls=ls,checkpoint_path=checkpoint_path)
            dump(folder/'result.json',result)
            row.update(status=result.status,requests=surface.requests,minima=len(result.minima),records=len(result.records),
                       record_statuses=[r.status for r in result.records],accepted=[r.accepted for r in result.records],
                       rotation_failed=sum(any(e.get('status')=='rotation_failed' for e in r.climb) for r in result.records),
                       biased_quench_failed=sum(any(e.get('status')=='biased_quench_failed' for e in r.climb) for r in result.records))
            for i,m in enumerate(result.minima[:21]):
                if time.monotonic()-started>=60 or time.monotonic()-started_all>=args.total_wall_seconds:
                    raise TimeoutError('wall cap before fresh evaluation')
                try:
                    fs=ASESurface(calc()); e,f=fs.evaluate(m.atoms); fresh.append(dict(index=i,energy=e,energy_error=e-m.energy,fmax=float(np.linalg.norm(f,axis=1).max()),
                        composition_match=bool(np.array_equal(m.atoms.numbers, initial.numbers)),
                        cell_unchanged=bool(np.array_equal(m.atoms.cell.array, initial.cell.array)),
                        pbc_unchanged=bool(np.array_equal(m.atoms.pbc, initial.pbc)),
                        qualified=bool(np.linalg.norm(f,axis=1).max()<=.03)))
                except TimeoutError:
                    raise
                except Exception as exc: fresh.append(dict(index=i,error=repr(exc)))
                dump(folder/'fresh.json',fresh)
            for check in fresh:
                if 'error' not in check:
                    check['qualified'] = bool(check['qualified'] and np.isfinite(check['energy']) and
                        np.isfinite(check['energy_error']) and abs(check['energy_error']) <= 1e-7 and
                        check['composition_match'] and check['cell_unchanged'] and check['pbc_unchanged'])
            dump(folder/'fresh.json',fresh); row['fresh']=fresh
        except Exception as exc: row.update(status='exception',error=repr(exc),requests=surface.requests)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, old_alarm_handler)
        row.update(denied=denied,wall_seconds=time.monotonic()-started); dump(folder/'summary.json',row); summaries.append(row); dump(out/'summary.json',summaries)
        print(name,variant,seed,row['status'],row.get('requests'),flush=True)
    dump(out/'summary.json',summaries)


if __name__=='__main__': main()
