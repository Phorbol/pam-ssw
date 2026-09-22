#!/usr/bin/env python3
"""Prepared LS plus observation-pool restart qualification; execution is opt-in."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next((parent for parent in HERE.parents if (parent / 'pamssw').is_dir()), None)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.generic,)): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if hasattr(value, '__dict__'): return jsonable(vars(value))
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


class CountedSurface:
    def __init__(self, calculator, ledger, cap, wall):
        self.calculator, self.ledger, self.cap, self.wall = calculator, Path(ledger), cap, wall
        self.requests = self.denials = 0
        self.started = time.monotonic()
        self.boundary = None

    def evaluate(self, atoms):
        if self.requests >= self.cap or time.monotonic() - self.started >= self.wall:
            self.denials += 1
            self.boundary = 'request_cap' if self.requests >= self.cap else 'wall_cap'
            raise RuntimeError(self.boundary)
        self.requests += 1
        work = atoms.copy(); work.calc = self.calculator
        energy = float(work.get_potential_energy())
        forces = np.asarray(work.get_forces(), float)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError('nonfinite energy/forces')
        with self.ledger.open('a') as stream:
            stream.write(json.dumps({
                'request': self.requests, 'energy_eV': energy,
                'fmax_eV_A': float(np.linalg.norm(forces, axis=1).max()),
            }) + '\n')
        return energy, forces


class PoolSelector:
    """Qualified-observation pool with transparent or deterministic policy."""
    def __init__(self, mode):
        self.mode = mode
        self.decisions = []

    def __call__(self, snapshot, selector_rng):
        observations = snapshot.observations
        current = snapshot.current_index
        chosen = None
        if self.mode == 'deterministic_other_index' and len(observations) >= 2:
            chosen = (current + 1) % len(observations)
            if chosen == current:
                chosen = None
        self.decisions.append({
            'step': snapshot.step, 'observation_count': len(observations),
            'current_index': current, 'chosen_index': chosen,
            'restarted': chosen is not None and chosen != current,
            'pool_cost_requests': snapshot.cost,
        })
        return chosen


def atoms_from_json(payload):
    from ase import Atoms
    return Atoms(numbers=payload['numbers'], positions=payload['positions'],
                 cell=payload['cell'], pbc=payload['pbc'])


def prepare(out):
    if REPO is None:
        raise RuntimeError('prepare must run from the integration checkout')
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((HERE / 'plan.json').read_text())
    shutil.copytree(REPO / 'pamssw', out / 'source' / 'pamssw',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (out / 'source-manifest.json').write_text(json.dumps({
        str(path.relative_to(out / 'source')): sha256(path)
        for path in (out / 'source').rglob('*.py')}, indent=2) + '\n')
    plan['source_commit_at_prepare'] = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    plan['source_status_at_prepare'] = subprocess.check_output(
        ['git', '-C', str(REPO), 'status', '--short'], text=True)
    (out / 'inputs').mkdir()
    for name, spec in plan['cases'].items():
        source = REPO / spec['input_source']
        if sha256(source) != spec['input_sha256']:
            raise ValueError(f'input changed: {name}')
        target = out / 'inputs' / (name + source.suffix)
        shutil.copy2(source, target)
        spec['prepared_path'] = str(target.relative_to(out))
        spec['prepared_sha256'] = sha256(target)
        config_source = REPO / spec['config_source']
        if not config_source.exists():
            raise FileNotFoundError(config_source)
        config_target = out / 'config-sources' / (name + '.json')
        config_target.parent.mkdir(exist_ok=True)
        shutil.copy2(config_source, config_target)
        spec['config_prepared_path'] = str(config_target.relative_to(out))
        spec['config_sha256'] = sha256(config_target)
    dump(out / 'plan.json', plan)
    shutil.copy2(HERE / 'runner.py', out / 'runner.py')


def make_ls(spec, payload):
    from pamssw.standalone import LSPrequenchSettings, NativeLSSettings
    values = payload.get('native_ls', payload.get('ls'))
    if values is None:
        values = payload['ls']
    def pairs(raw):
        return {tuple(int(part.strip(' ()')) for part in key.split(',')): value
                for key, value in raw.items()}
    values = dict(values)
    values['bond_energies'] = pairs(values['bond_energies'])
    values['bond_lengths'] = pairs(values['bond_lengths'])
    if values.get('prequench'):
        values['prequench'] = LSPrequenchSettings(**values['prequench'])
    values.pop('parameter_source', None)
    for key in ('energy_filter', 'length_filter'):
        if values.get(key) is not None:
            values[key] = pairs(values[key])
    return NativeLSSettings(**values)


def run_one(out, plan, name, arm):
    from ase.io import read
    import torch
    from mace.calculators import MACECalculator
    from pamssw.standalone import NativeMCSettings, SSWConfig, run_ssw
    spec = plan['cases'][name]
    path = out / spec['prepared_path']
    if sha256(path) != spec['prepared_sha256']:
        raise ValueError(f'prepared input changed: {name}')
    if path.suffix == '.traj':
        atoms = read(path)
    else:
        atoms = atoms_from_json(json.loads(path.read_text())['atoms'])
    config_payload = json.loads((out / spec['config_prepared_path']).read_text())
    config = SSWConfig(**config_payload['config'])
    torch.set_num_threads(1); torch.manual_seed(plan['runtime']['torch_manual_seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if sha256(spec['model']) != spec['model_sha256']:
        raise ValueError(f'model changed: {name}')
    calculator = MACECalculator(model_paths=spec['model'], head=spec['head'], device='cuda',
                                default_dtype=plan['runtime']['dtype'], enable_cueq=False,
                                enable_oeq=False)
    folder = out / f'{name}-{arm}'; folder.mkdir()
    surface = CountedSurface(calculator, folder / 'search.jsonl',
                             plan['search_cap_per_arm'], plan['wall_seconds_per_arm'])
    selector = PoolSelector(arm)
    row = {'case': name, 'arm': arm, 'status': 'started', 'restart_reached': False}
    try:
        kwargs = dict(atoms=atoms, surface=surface, steps=plan['outer_steps'], config=config,
                      rng=np.random.default_rng(spec['seed']), ls=make_ls(spec, config_payload),
                      starter_selector=selector,
                      selector_rng=np.random.default_rng(spec['seed'] + 1000003))
        if name.startswith('c60'):
            kwargs['mc'] = NativeMCSettings(energy_tol=config_payload['native_mc']['energy_tol_eV'],
                                              maxtrap=config_payload['native_mc']['maxtrap'])
        result = run_ssw(**kwargs)
        with (folder / 'result.pkl').open('wb') as stream:
            pickle.dump(result, stream, protocol=4)
        committed = [getattr(record, 'starter_selection', None)
                     for record in result.records]
        committed = [selection for selection in committed if selection is not None]
        restarts = [selection for selection in committed if selection.get('restarted', False)]
        fresh = []
        fresh_calc = MACECalculator(model_paths=spec['model'], head=spec['head'], device='cuda',
                                    default_dtype=plan['runtime']['dtype'], enable_cueq=False,
                                    enable_oeq=False)
        fresh_surface = CountedSurface(fresh_calc, folder / 'fresh.jsonl',
                                       plan['fresh_cap_per_arm'], plan['wall_seconds_per_arm'])
        for index, minimum in enumerate(result.minima[:plan['fresh_checks']['per_arm_maxima']]):
            try:
                energy, forces = fresh_surface.evaluate(minimum.atoms)
                fresh.append({'index': index, 'energy_eV': energy,
                              'energy_error_eV': energy - minimum.energy,
                              'fmax_eV_A': float(np.linalg.norm(forces, axis=1).max()),
                              'force_threshold_eV_A': config.fmax,
                              'force_qualified': bool(np.linalg.norm(forces, axis=1).max() <= config.fmax),
                              'finite_energy': True, 'finite_forces': True,
                              'composition_unchanged': bool(np.array_equal(minimum.atoms.numbers, atoms.numbers)),
                              'cell_unchanged': bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array)),
                              'pbc_unchanged': bool(np.array_equal(minimum.atoms.pbc, atoms.pbc))})
            except Exception as error:
                fresh.append({'index': index, 'status': 'error', 'error': repr(error)})
        row.update(status=result.status, search_requests=surface.requests,
                   minima=len(result.minima), restart_reached=bool(restarts),
                   restart_count=len(restarts), requested_selections=selector.decisions,
                   committed_selections=committed,
                   ls_preparation=[None if r.ls_preparation is None else
                       {k: v for k, v in r.ls_preparation.items() if k != 'soft_quench'}
                       for r in result.records],
                   ls_updates=[getattr(r, 'ls_update', None) for r in result.records],
                   record_statuses=[r.status for r in result.records], fresh=fresh,
                   fresh_requests=fresh_surface.requests, total_ef=surface.requests + fresh_surface.requests)
    except Exception as error:
        row.update(status='exception', error=repr(error), requested_selections=selector.decisions,
                   search_requests=surface.requests, total_ef=surface.requests)
    row.update(boundary=surface.boundary, denials=surface.denials,
               execution_host=os.uname().nodename)
    dump(folder / 'summary.json', row)
    return row


def execute(out):
    plan = json.loads((out / 'plan.json').read_text())
    if (out / 'execution-started.json').exists():
        raise FileExistsError('preserve prior attempt')
    dump(out / 'execution-started.json', {'job_id': os.getenv('SLURM_JOB_ID'), 'host': os.uname().nodename})
    sys.path.insert(0, str(out / 'source'))
    manifest = json.loads((out / 'source-manifest.json').read_text())
    for relative, expected in manifest.items():
        if sha256(out / 'source' / relative) != expected:
            raise ValueError(f'frozen source changed: {relative}')
    rows = []
    for name in plan['cases']:
        for arm in plan['arms']:
            rows.append(run_one(out, plan, name, arm))
            dump(out / 'summary.json', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error('choose exactly one of --prepare/--execute')
    (prepare if args.prepare else execute)(args.output.resolve())
