"""Prepare/execute the bounded fixed-cell TYPE1 periodic-GA comparison.

The default command only creates an immutable source/input snapshot and makes
no calculator request.  Execution is deliberately explicit and runs the
copied runner from that snapshot.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = ROOT / 'research/ga_ssw/evidence'
CU_SOURCE = RESEARCH_ROOT / 'stage-control-e2e-20260912/cu31_fixed-pam_height_width-baseline-seed11/raw-result.json'
AL_SOURCE = RESEARCH_ROOT / 'native-ls-periodic-multicase-20260912/Al31-native-mic/result.json'


def _atoms_from_json(d):
    from ase import Atoms
    return Atoms(numbers=d['numbers'], positions=d['positions'],
                 cell=d['cell'], pbc=d['pbc'])


def _extract(path, *, nested=False):
    import numpy as np
    data = json.loads(path.read_text())
    raw = data['raw_result'] if nested else data
    minima = raw['minima'][:3]
    for item in minima:
        reconstructed = _atoms_from_json(item['atoms'])
        if 'masses' in item['atoms'] and not np.allclose(
                reconstructed.get_masses(), item['atoms']['masses'], rtol=0., atol=1e-12):
            raise ValueError(f'{path}: non-default isotope masses are outside TYPE1 runner contract')
    return [_atoms_from_json(x['atoms']) for x in minima], [float(x['energy']) for x in minima]


def _prepare(out, *, full_walk=False, seed=7):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = out / 'source'
    shutil.copytree(ROOT / 'pamssw', source / 'pamssw',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    cu, cu_e = _extract(CU_SOURCE)
    al, al_e = _extract(AL_SOURCE, nested=True)
    (out / 'inputs').mkdir()
    from ase.io import write
    write(out / 'inputs/cu31-parents.extxyz', cu)
    write(out / 'inputs/al31-parents.extxyz', al)
    protocol = ROOT / 'docs/research/2026-09-12-fixed-periodic-ga-candidate-protocol.md'
    shutil.copy2(protocol, out / 'protocol.md')
    plan = {
        'cases': {
            'Cu31': {'model': 'ASE EMT', 'parents': str(out/'inputs/cu31-parents.extxyz'),
                     'source': str(CU_SOURCE), 'source_kind': 'stage-control raw-result minima'},
            'Al31': {'model': 'ASE EMT', 'parents': str(out/'inputs/al31-parents.extxyz'),
                     'source': str(AL_SOURCE), 'source_kind': 'native-LS raw_result minima'},
        },
        'seed': int(seed), 'walk_mode': 'full' if full_walk else 'bounded',
        'request_cap': 4000, 'wall_seconds': 60,
        'ga_config': {'quick_steps': 1 if full_walk else 0, 'generations': 1,
                      'generation_steps': 1 if full_walk else 0,
                      'fine_steps': 1, 'regions': 3, 'fine_regions': 1, 'min_ga': 1,
                      'max_batches': 1, 'max_cut_attempts': 30, 'max_pair_attempts': 50,
                      'partition_max_draws': 100, 'slots_per_parent': 1,
                      'cuts_per_slot': 1},
        'ssw_config': {'width': .1, 'rotation_bias': 100., 'max_gaussians': 14,
                       'temperature_K': 150., 'fmax': .01, 'relax_steps': 200,
                       'fd_step': 1e-4, 'rotation_hvp': 100, 'rotation_tol': .02,
                       'direction_sampling': 'global', 'rotation_solver': 'dimer',
                       'cluster_frame': 'translation_only',
                       'quench_optimizer': 'safe-lbfgs-total', 'lbfgs_memory': 10,
                       'bias_fmax': .1,
                       'fixed_cell': True, 'bias_quench_adapter': None,
                       'LS': None},
        'routing': {'bond_lengths_A': {'Cu-Cu': 2.6, 'Al-Al': 2.9},
                    'neighbor_range': 1.1, 'bond_limits_A': {'same_element': .5},
                    'projection_weights': [.3, .2, .2, .1, .1, .1],
                    'identity': {'backend': 'pymatgen StructureMatcher',
                                 'ltol': .2, 'stol': .3, 'angle_tol': 5.,
                                 'scale': False, 'primitive_cell': True,
                                 'attempt_supercell': True}},
        'note': 'fixed-cell TYPE1 wiring and observed offspring/fine boundary only; no basin or efficiency claim',
    }
    manifest = {
        'source_root': str(source),
        'source_files': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source.rglob('*.py')},
        'candidate_sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (CU_SOURCE, AL_SOURCE)},
        'input_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (out/'inputs').glob('*.extxyz')},
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'execution': 'not executed; preparation only',
    }
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (out/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'candidate-provenance.json').write_text(json.dumps({
        'Cu31': {'source': str(CU_SOURCE), 'energies_eV': cu_e,
                 'mass_contract': 'saved masses match ASE defaults; reconstruction uses numbers only',
                 'composition': cu[0].get_chemical_formula(), 'cell': cu[0].cell.array.tolist(), 'pbc': cu[0].pbc.tolist()},
        'Al31': {'source': str(AL_SOURCE), 'energies_eV': al_e,
                 'mass_contract': 'saved masses match ASE defaults; reconstruction uses numbers only',
                 'composition': al[0].get_chemical_formula(), 'cell': al[0].cell.array.tolist(), 'pbc': al[0].pbc.tolist()},
    }, indent=2)+'\n')
    shutil.copy2(Path(__file__), out/'runner.py')


def _child(out):
    import importlib, importlib.metadata
    import numpy as np
    from ase import __version__ as ase_version
    from ase.io import read
    from ase.calculators.emt import EMT
    import pamssw
    source = (out/'source').resolve()
    assert source in Path(pamssw.__file__).resolve().parents
    from pamssw.standalone import ASESurface, SSWConfig
    from pamssw.standalone.periodic_ga_reference import PeriodicGAConfig, run_periodic_ga, pymatgen_identity
    periodic_module = Path(importlib.import_module('pamssw.standalone.periodic_ga_reference').__file__).resolve()
    assert source in periodic_module.parents, periodic_module
    plan = json.loads((out/'plan.json').read_text())
    (out/'runtime-import.json').write_text(json.dumps({'pamssw': str(Path(pamssw.__file__).resolve()),
        'periodic_ga_reference': str(periodic_module)}, indent=2)+'\n')
    try: scipy_version = importlib.metadata.version('scipy')
    except importlib.metadata.PackageNotFoundError: scipy_version = 'unknown'
    try: pymatgen_version = importlib.metadata.version('pymatgen')
    except importlib.metadata.PackageNotFoundError: pymatgen_version = 'unknown'
    (out/'runtime-versions.json').write_text(json.dumps({'python': sys.version,
        'numpy': np.__version__, 'scipy': scipy_version, 'ase': ase_version,
        'pymatgen': pymatgen_version, 'pamssw_file': str(Path(pamssw.__file__).resolve()),
        'PYTHONPATH': os.environ.get('PYTHONPATH','')}, indent=2)+'\n')

    def dump(x):
        from dataclasses import fields, is_dataclass
        from ase import Atoms
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Atoms): return {'numbers': x.numbers.tolist(), 'positions': x.positions.tolist(),
            'masses': x.get_masses().tolist(), 'cell': x.cell.array.tolist(), 'pbc': x.pbc.tolist()}
        if is_dataclass(x): return {f.name: dump(getattr(x,f.name)) for f in fields(x)}
        if isinstance(x, dict): return {str(k): dump(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [dump(v) for v in x]
        return x

    def config():
        s=plan['ssw_config']; return SSWConfig(**{k:v for k,v in s.items() if k not in ('fixed_cell','bias_quench_adapter','LS')})
    def ga_config(): return PeriodicGAConfig(**plan['ga_config'])
    def fresh(obs, case):
        a=obs['atoms'].copy(); a.calc=None
        surface=ASESurface(EMT())
        try:
            e,f=surface.evaluate(a.copy())
        except Exception as exc:
            return {'error':repr(exc), 'requests':surface.requests}
        return {'energy': float(e), 'stored_energy': float(obs['energy']),
            'energy_error': float(e-obs['energy']), 'forces': f.tolist(),
            'fmax': float(np.linalg.norm(f,axis=1).max()),
            'force_qualified': bool(np.linalg.norm(f,axis=1).max() <= config().fmax),
            'cell_exact': bool(np.array_equal(a.cell.array, inputs[case][0].cell.array)),
            'composition_exact': bool(np.array_equal(np.sort(a.numbers), np.sort(inputs[case][0].numbers))),
            'pbc_exact': bool(np.array_equal(a.pbc, inputs[case][0].pbc)), 'requests': surface.requests}

    inputs={case: read(spec['parents'], index=':') for case,spec in plan['cases'].items()}
    for case, spec in plan['cases'].items():
        bond={(29,29):2.6} if case=='Cu31' else {(13,13):2.9}
        base_matcher=pymatgen_identity(ltol=.2,stol=.3,angle_tol=5.)
        matcher_calls=[]
        def matcher(a,b):
            call={'call':len(matcher_calls)+1, 'lhs_positions':a.positions.tolist(),
                  'rhs_positions':b.positions.tolist(), 'cell':a.cell.array.tolist(),
                  'pbc':a.pbc.tolist()}
            try:
                call['matched']=bool(base_matcher(a,b))
            except Exception as exc:
                call['error']=repr(exc); matcher_calls.append(call); raise
            matcher_calls.append(call)
            return call['matched']
        folder=out/case; folder.mkdir()
        ledger=folder/'evaluations.jsonl'; start=time.monotonic()
        class Counted(ASESurface):
            exhausted=False
            def evaluate(self, atoms):
                before=self.requests; row={'request': before+1, 'atoms': dump(atoms)}
                if before>=plan['request_cap'] or time.monotonic()-start>=plan['wall_seconds']:
                    row.update(kind='denied', charged=False, error='request_cap' if before>=plan['request_cap'] else 'wall_cap')
                    with ledger.open('a') as h: h.write(json.dumps(row)+'\n')
                    self.exhausted=True; raise RuntimeError(row['error'])
                try:
                    e,f=super().evaluate(atoms); row.update(kind='paid', energy=e, forces=f, charged=self.requests>before)
                except Exception as error:
                    row.update(kind='failure', error=repr(error), charged=self.requests>before)
                    with ledger.open('a') as h: h.write(json.dumps(dump(row))+'\n')
                    raise
                with ledger.open('a') as h: h.write(json.dumps(dump(row))+'\n')
                return e,f
        surface=Counted(EMT())
        meta={'case':case,'model':spec['model'],'seed':plan['seed'],'ga_config':plan['ga_config'],
              'ssw_config':dump(config()), 'ssw_config_requested':plan['ssw_config'],
              'routing':plan['routing'],'parents':dump(inputs[case]),
              'source':spec['source']}
        (folder/'arm-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        result=None; error=None
        try:
            result=run_periodic_ga(inputs[case], surface, config=ga_config(), walker_config=config(),
                rng=np.random.default_rng(plan['seed']), descriptor_basis=inputs[case],
                bond_lengths=bond, neighbor_range=1.1, projection_weights=plan['routing']['projection_weights'],
                bond_limits={(29,29):.5} if case=='Cu31' else {(13,13):.5}, matcher=matcher, fixed_cell=True)
        except Exception as exc: error=repr(exc)
        search_wall=time.monotonic()-start
        (folder/'raw-result.json').write_text(json.dumps(dump(result),indent=2)+'\n' if result is not None else 'null\n')
        fresh_start=time.monotonic()
        fresh_rows=[]
        if result is not None:
            for obs in result['observations']:
                fresh_rows.append({'observation_id':obs['id'], **fresh(obs,case)})
        (folder/'matcher-calls.json').write_text(json.dumps(dump(matcher_calls),indent=2)+'\n')
        (folder/'fresh-observations.json').write_text(json.dumps(fresh_rows,indent=2)+'\n')
        fresh_wall=time.monotonic()-fresh_start
        rows=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
        search_requests=sum(bool(x.get('charged')) for x in rows)
        fresh_requests=sum(int(x.get('requests',0)) for x in fresh_rows)
        stage_audit=[]
        if result is not None:
            for walk in result['walks']:
                walk_result=walk.get('result')
                stage_audit.append({'phase':walk['phase'], 'generation':walk['generation'],
                    'walk_requests':walk['requests'],
                    'result_requests':None if walk_result is None else walk_result.evaluation_requests,
                    'record_count':None if walk_result is None else len(walk_result.records)})
        summary={'case':case,'status':None if result is None else result['status'],'error':error,
                 'search_requests':search_requests,'surface_requests':surface.requests,
                 'fresh_requests':fresh_requests,'ledger_events':len(rows),
                 'observations':0 if result is None else len(result['observations']),
                 'archive_representatives':0 if result is None else len(result['archive']),
                 'stage_audit':stage_audit,
                 'fresh_checks':len(fresh_rows),'fresh_qualified':bool(fresh_rows) and all(x.get('force_qualified',False) for x in fresh_rows),
                 'requests_reconciled': (search_requests == surface.requests and
                    (result is None or surface.requests == result['requests'])),
                 'search_wall_seconds':search_wall,'fresh_wall_seconds':fresh_wall,
                 'result_available':result is not None,'wall_seconds':time.monotonic()-start}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--execute',action='store_true'); ap.add_argument('--child',action='store_true'); ap.add_argument('--full-walk',action='store_true'); ap.add_argument('--seed',type=int,default=7); a=ap.parse_args(); out=a.output.resolve()
    if a.child: _child(out); return
    if a.execute:
        if not (out/'runner.py').exists(): raise FileNotFoundError('prepare first')
        old=os.environ.get('PYTHONPATH',''); paths=[str(out/'source'),'/tmp/pam-ssw-tblite-20260909','.']+([old] if old else [])
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=os.pathsep.join(paths),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        subprocess.run([sys.executable,str(out/'runner.py'),'--child','--output',str(out)],check=True,env=env); return
    _prepare(out, full_walk=a.full_walk, seed=a.seed); print(json.dumps({'status':'prepared','output':str(out),'execute':False,'full_walk':a.full_walk,'seed':a.seed}))

if __name__=='__main__': main()
