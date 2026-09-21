"""Prepare (and, after review, execute) the frozen 12-arm stage-control run.

Preparation performs no calculator request.  Execution is delegated to this
file copied into the output directory, with all imports resolved from the
frozen source tree.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'research/ga_ssw/evidence/ase-basin-hopping-baseline-20260912-v2'
RESEARCH_FILES = ('native_stage_predicate.py', 'native_stage_quench.py',
                  'native_stage_quench_adapter.py')


def _prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = out / 'source'
    shutil.copytree(ROOT / 'pamssw', source / 'pamssw',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'research/ga_ssw').mkdir(parents=True)
    for name in RESEARCH_FILES:
        shutil.copy2(ROOT / 'research/ga_ssw' / name, source / 'research/ga_ssw' / name)
    (out / 'inputs').mkdir()
    for name in ('cu13', 'cu31_fixed', 'bicyclobutane'):
        shutil.copy2(FIXTURE / f'{name}.extxyz', out / 'inputs' / f'{name}.extxyz')
    protocol = ROOT / 'docs/research/2026-09-12-stage-control-e2e-protocol.md'
    shutil.copy2(protocol, out / 'protocol.md')
    manifest = {
        'source_root': str(source),
        'source_files': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source.rglob('*.py')},
        'inputs': {name: hashlib.sha256((out/'inputs'/f'{name}.extxyz').read_bytes()).hexdigest()
                   for name in ('cu13','cu31_fixed','bicyclobutane')},
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'execution': 'not executed; preparation only',
    }
    (out / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    plan = {
        'cases': {'cu13': 'EMT', 'cu31_fixed': 'EMT', 'bicyclobutane': 'GFN2-xTB'},
        'gaussian_strategies': ['forward_force', 'pam_height_width'],
        'arms_per_strategy': ['baseline', 'stage_control'],
        'seed': 11, 'outer_attempts': 2, 'request_cap': 4000, 'wall_seconds': 60,
        'shared': {'width_A': .1, 'max_gaussians': 14, 'temperature_K': 150.,
                   'fd_step_A': 1e-4, 'rotation_bias': 100., 'rotation_hvp': 100,
                   'rotation_tol': .02, 'optimizer': 'safe-lbfgs-total',
                   'lbfgs_memory': 10, 'relax_steps': 200, 'bias_fmax': .1,
                   'fmax': .01, 'direction_sampling': 'global'},
        'rotation_solver': 'dimer',
        'frame_policy': {'pbc': 'translation_only', 'nonperiodic': 'cartesian',
                         'reason': 'public run_ssw rejects translation_only for non-PBC inputs'},
        'stage_control': {'budget_initial': 200, 'budget_later': 200,
                          'counter_start': 1, 'climb_stopf': .1 / (3 ** .5),
                          'e_maxlimit': 100., 'f_maxlimit': 100.,
                          'e_maxlimit_gm': 100., 'stop_on': 'known_stop',
                          'energy_reference': 'current', 'gm_reference': 'best'},
        'source_inputs': {name: str(out/'inputs'/f'{name}.extxyz')
                          for name in ('cu13','cu31_fixed','bicyclobutane')},
        'note': 'Experimental stage-control diagnostic; no native parity or efficiency claim.',
    }
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    shutil.copy2(Path(__file__), out / 'runner.py')


def _child(out):
    import numpy as np
    from ase import __version__ as ase_version
    from ase.io import read
    from ase.calculators.emt import EMT
    from tblite.ase import TBLite
    import importlib
    import pamssw
    source = out / 'source'
    assert source in Path(pamssw.__file__).resolve().parents
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
    from research.ga_ssw.native_stage_quench_adapter import StatefulNativeStageAdapter
    imported_research = {n: str(Path(importlib.import_module('research.ga_ssw.' + n[:-3]).__file__).resolve())
                         for n in RESEARCH_FILES}
    expected_research = {n: (source / 'research/ga_ssw' / n).resolve()
                         for n in RESEARCH_FILES}
    assert all(Path(imported_research[n]).resolve() == expected_research[n]
               for n in RESEARCH_FILES), (imported_research, expected_research)
    (out / 'runtime-import.json').write_text(json.dumps({
        'pamssw': str(Path(pamssw.__file__).resolve()), 'ase': ase_version,
        'research_modules': imported_research}, indent=2) + '\n')
    plan = json.loads((out/'plan.json').read_text())
    from importlib.metadata import version, PackageNotFoundError
    try: tblite_version = version('tblite')
    except PackageNotFoundError: tblite_version = 'unknown'
    try: scipy_version = version('scipy')
    except PackageNotFoundError: scipy_version = 'unknown'
    (out/'runtime-versions.json').write_text(json.dumps({
        'python': sys.version, 'numpy': np.__version__, 'scipy': scipy_version,
        'ase': ase_version, 'tblite': tblite_version,
        'pamssw_file': str(Path(pamssw.__file__).resolve()),
        'PYTHONPATH': os.environ.get('PYTHONPATH', '')}, indent=2) + '\n')
    # All metadata is loaded and written before constructing any calculator.
    inputs = {k: read(v) for k, v in plan['source_inputs'].items()}
    (out/'inputs-metadata.json').write_text(json.dumps({k: {
        'numbers': a.numbers.tolist(), 'pbc': a.pbc.tolist(),
        'cell': a.cell.array.tolist(), 'positions': a.positions.tolist()}
        for k, a in inputs.items()}, indent=2) + '\n')

    def dump(value):
        from dataclasses import is_dataclass
        from ase import Atoms
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, Atoms):
            return {'numbers': value.numbers.tolist(), 'positions': value.positions.tolist(),
                    'masses': value.get_masses().tolist(),
                    'cell': value.cell.array.tolist(), 'pbc': value.pbc.tolist()}
        if is_dataclass(value):
            return {f.name: dump(getattr(value, f.name)) for f in __import__('dataclasses').fields(value)}
        if isinstance(value, dict): return {str(k): dump(v) for k,v in value.items()}
        if isinstance(value, (tuple, list)): return [dump(v) for v in value]
        return value

    def make_config(a):
        return SSWConfig(width=.1, rotation_bias=100., max_gaussians=14,
            temperature_K=150., fmax=.01, relax_steps=200, fd_step=1e-4,
            rotation_hvp=100, rotation_tol=.02, direction_sampling='global',
            rotation_solver='dimer', bias_fmax=.1,
            cluster_frame='translation_only' if bool(a.pbc.all()) else 'cartesian',
            quench_optimizer='safe-lbfgs-total', lbfgs_memory=10)

    def calculator(name):
        return TBLite(method='GFN2-xTB', accuracy=.001, verbosity=0) if name == 'GFN2-xTB' else EMT()

    def connectivity(atoms):
        from ase.data import covalent_radii
        edges = {i: set() for i in range(len(atoms))}
        for i in range(len(atoms)):
            for j in range(i):
                d = atoms.get_distance(i, j, mic=bool(atoms.pbc.all()))
                if d < 1.25 * (covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]]):
                    edges[i].add(j); edges[j].add(i)
        seen = set(); sizes = []
        for i in edges:
            if i in seen: continue
            stack=[i]; seen.add(i); n=0
            while stack:
                k=stack.pop(); n += 1
                for j in edges[k]-seen: seen.add(j); stack.append(j)
            sizes.append(n)
        return sorted(sizes)

    for case, model in plan['cases'].items():
        for strategy in plan['gaussian_strategies']:
            for arm in ('baseline', 'stage_control'):
                folder = out / f'{case}-{strategy}-{arm}-seed11'; folder.mkdir()
                ledger = folder/'evaluations.jsonl'; started = time.monotonic(); result = None
                class Counted(ASESurface):
                    def evaluate(self, atoms):
                        row = {'kind': 'search', 'request': self.requests + 1,
                               'atoms': dump(atoms)}
                        if self.requests >= plan['request_cap'] or time.monotonic()-started >= plan['wall_seconds']:
                            row.update(kind='denied', error=('request_cap' if self.requests >= plan['request_cap'] else 'wall_cap'), charged=False)
                            with ledger.open('a') as handle:
                                handle.write(json.dumps(dump(row))+'\n')
                            raise RuntimeError(row['error'])
                        try:
                            before = self.requests
                            e, f = super().evaluate(atoms)
                            row.update(energy=e, forces=f, charged=self.requests > before)
                        except Exception as exc:
                            row['charged'] = self.requests > before
                            row.update(kind='failure', error=repr(exc))
                            with ledger.open('a') as handle:
                                handle.write(json.dumps(dump(row))+'\n')
                            raise
                        with ledger.open('a') as handle:
                            handle.write(json.dumps(dump(row))+'\n')
                        return e, f
                surface = Counted(calculator(model)); config = make_config(inputs[case])
                gaussian = None if strategy == 'forward_force' else PAMCurvatureGaussian(mode='height_width')
                adapter = None
                if arm == 'stage_control':
                    adapter = StatefulNativeStageAdapter(
                        predicate_kwargs={'climb_stopf': .1/(3**.5), 'maxe_height': 0.,
                            'e_maxlimit': 100., 'f_maxlimit': 100., 'e_maxlimit_gm': 100.,
                            'ngaus_relax': 200, 'ngaus_relax_ini': 200,
                            'multi_pes': False, 'counter_start': 1,
                            'energy_margin': .1, 'saved_energy_margin': 1.},
                        stop_on='known_stop', energy_reference='current', gm_reference='best')
                arm_metadata = {
                    'case': case, 'model': model, 'strategy': strategy,
                    'arm': arm, 'seed': 11, 'input': dump(inputs[case]),
                    'config': dump(config),
                    'gaussian_policy': None if gaussian is None else gaussian.parameters(),
                    'adapter': None if adapter is None else {
                        'predicate_kwargs': dump(adapter.predicate_kwargs),
                        'stop_on': adapter.stop_on,
                        'energy_reference': adapter.energy_reference,
                        'gm_reference': adapter.gm_reference,
                    },
                    'budget': {'request_cap': plan['request_cap'],
                               'wall_seconds': plan['wall_seconds']},
                }
                (folder/'arm-metadata.json').write_text(json.dumps(dump(arm_metadata), indent=2) + '\n')
                row = {'case': case, 'model': model, 'strategy': strategy, 'arm': arm,
                       'seed': 11, 'status': None, 'requests': 0, 'error': None}
                try:
                    result = run_ssw(inputs[case].copy(), surface, steps=2, config=config,
                        rng=np.random.default_rng(11), gaussian_policy=gaussian,
                        bias_quench_adapter=adapter)
                    row.update(status=result.status, requests=surface.requests,
                               minima=len(result.minima))
                except Exception as exc:
                    row.update(status='exception', requests=surface.requests, error=repr(exc),
                               partial=dump(result) if result is not None else None)
                search_wall = time.monotonic() - started
                (folder/'raw-result.json').write_text(json.dumps(dump(result), indent=2)+'\n' if result is not None else 'null\n')
                fresh_checks=[]; fresh_requests=0; fresh_wall=0.0
                if result is not None:
                    for i, minimum in enumerate(result.minima):
                        fresh_surface = None
                        fresh_started = time.monotonic()
                        try:
                            fresh_surface = ASESurface(calculator(model))
                            e, forces = fresh_surface.evaluate(minimum.atoms.copy())
                            fmax = float(np.linalg.norm(forces, axis=1).max())
                            fresh_checks.append({'index': i, 'energy': float(e),
                                'energy_error': float(e-minimum.energy), 'fmax': fmax,
                                'forces': forces.tolist(),
                                'force_qualified': bool(fmax <= config.fmax),
                                'cell_exact': bool(np.array_equal(minimum.atoms.cell.array, inputs[case].cell.array)),
                                'pbc_exact': bool(np.array_equal(minimum.atoms.pbc, inputs[case].pbc)),
                                'connectivity': connectivity(minimum.atoms)})
                        except Exception as exc:
                            fresh_checks.append({'index': i, 'error': repr(exc), 'requests': getattr(fresh_surface, 'requests', 0)})
                        finally:
                            fresh_wall += time.monotonic() - fresh_started
                        fresh_requests += getattr(fresh_surface, 'requests', 0)
                (folder/'fresh-checks.json').write_text(json.dumps(dump(fresh_checks), indent=2)+'\n')
                row['wall_seconds'] = search_wall
                row['fresh'] = {'requests': fresh_requests, 'wall_seconds': fresh_wall,
                                'checks': len(fresh_checks),
                                'all_force_qualified': bool(fresh_checks) and all(x.get('force_qualified', False) for x in fresh_checks)}
                row = {k: row.get(k) for k in ('case','model','strategy','arm','seed','status','requests','minima','error','wall_seconds','fresh')}
                (folder/'summary.json').write_text(json.dumps(dump(row), indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--child', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    if args.child:
        _child(out); return
    if args.execute:
        if not (out/'runner.py').exists():
            raise FileNotFoundError('prepared runner.py is required before --execute')
        existing = os.environ.get('PYTHONPATH', '')
        path_parts = [str(out/'source'), '/tmp/pam-ssw-tblite-20260909', '.']
        if existing:
            path_parts.append(existing)
        env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH=os.pathsep.join(path_parts))
        subprocess.run([sys.executable, str(out/'runner.py'), '--child', '--output', str(out)],
                       check=True, env=env)
        return
    _prepare(out)
    print(json.dumps({'status': 'prepared', 'output': str(out), 'execute': False}))


if __name__ == '__main__':
    main()
