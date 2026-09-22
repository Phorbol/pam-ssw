#!/usr/bin/env python3
"""Prepared MACE LS+PAM-pool checkpoint continuation runner; opt-in execution."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / 'pamssw').is_dir())
PRIOR = REPO / 'research/ga_ssw/evidence/ls-pool-restart-20260922/runner.py'
spec = importlib.util.spec_from_file_location('prior_ls_pool_runner', PRIOR)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    if hasattr(value, '__dict__'): return jsonable(vars(value))
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


class SharedBudget:
    def __init__(self, search_cap, fresh_cap, wall_seconds):
        self.search_cap, self.fresh_cap, self.wall_seconds = search_cap, fresh_cap, wall_seconds
        self.search_requests = self.fresh_requests = 0
        self.started = time.monotonic()


class SharedSurface:
    def __init__(self, calculator, ledger, budget, leg, kind='search'):
        self.calculator, self.ledger, self.budget, self.leg, self.kind = calculator, Path(ledger), budget, leg, kind
        self.requests = 0
        self.denials = 0
        self.boundary = None

    def evaluate(self, atoms):
        used = self.budget.search_requests if self.kind == 'search' else self.budget.fresh_requests
        cap = self.budget.search_cap if self.kind == 'search' else self.budget.fresh_cap
        if used >= cap or time.monotonic() - self.budget.started >= self.budget.wall_seconds:
            self.denials += 1
            self.boundary = 'request_cap' if used >= cap else 'wall_cap'
            with self.ledger.open('a') as stream:
                stream.write(json.dumps({'leg': self.leg, 'kind': self.kind,
                                         'status': 'denied', 'reason': self.boundary}) + '\n')
            raise RuntimeError(self.boundary)
        if self.kind == 'search': self.budget.search_requests += 1
        else: self.budget.fresh_requests += 1
        self.requests += 1
        work = atoms.copy(); work.calc = self.calculator
        try:
            energy = float(work.get_potential_energy())
            forces = np.asarray(work.get_forces(), float)
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError('nonfinite energy/forces')
            row = {'leg': self.leg, 'kind': self.kind, 'request': self.requests,
                   'cumulative_search': self.budget.search_requests,
                   'cumulative_fresh': self.budget.fresh_requests,
                   'energy_eV': energy, 'fmax_eV_A': float(np.linalg.norm(forces, axis=1).max())}
            with self.ledger.open('a') as stream: stream.write(json.dumps(row) + '\n')
            return energy, forces
        except Exception as error:
            with self.ledger.open('a') as stream:
                stream.write(json.dumps({'leg': self.leg, 'kind': self.kind,
                                         'status': 'error', 'error': repr(error),
                                         'cumulative_search': self.budget.search_requests,
                                         'cumulative_fresh': self.budget.fresh_requests}) + '\n')
            raise


def launch_provenance(out):
    patch_files = ('pamssw/standalone/paper_reference.py',
                   'pamssw/standalone/pool_checkpoint.py',
                   'research/ga_ssw/pool_starter_adapter.py')
    patch = subprocess.run(['git', '-C', str(REPO), 'diff', '--binary', 'HEAD', '--', *patch_files],
                           check=True, capture_output=True, text=False).stdout
    (out / 'code.patch').write_bytes(patch)
    dump(out / 'launch-provenance.json', {
        'host': os.uname().nodename, 'git_head': subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True),
        'patch_files': patch_files, 'runner': str(Path(__file__).resolve()),
    })


def make_case(plan, name):
    item = plan['cases'][name]
    root = REPO / item['prepared_root']
    input_path, config_path = root / item['input'], root / item['config']
    if sha256(input_path) != item['input_sha256'] or sha256(config_path) != item['config_sha256']:
        raise ValueError(f'prepared input/config changed: {name}')
    payload = json.loads(config_path.read_text())
    from ase.io import read
    atoms = (prior.atoms_from_json(json.loads(input_path.read_text())['atoms'])
             if input_path.suffix == '.json' else read(input_path))
    from pamssw.standalone import SSWConfig
    return item, atoms, payload, SSWConfig(**payload['config'])


def make_calculator(item):
    import torch
    from mace.calculators import MACECalculator
    if sha256(item['model']) != item['model_sha256']:
        raise ValueError('model changed')
    torch.set_num_threads(1); torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return MACECalculator(model_paths=item['model'], head=item['head'], device='cuda',
                          default_dtype='float64', enable_cueq=False, enable_oeq=False)


def record_result(folder, result, adapter, selector_rng, kernel_rng, surface):
    with (folder / 'result.pkl').open('wb') as stream: pickle.dump(result, stream, protocol=4)
    with (folder / 'pool-export.pkl').open('wb') as stream: pickle.dump(adapter.export_state(), stream, protocol=4)
    with (folder / 'selector-rng-state.pkl').open('wb') as stream: pickle.dump(
        copy.deepcopy(selector_rng.bit_generator.state), stream, protocol=4)
    with (folder / 'main-rng-state.pkl').open('wb') as stream: pickle.dump(
        copy.deepcopy(kernel_rng.bit_generator.state), stream, protocol=4)
    return {
        'status': result.status, 'evaluation_requests': result.evaluation_requests,
        'surface_requests': surface.requests, 'surface_denials': surface.denials,
        'records': [{'index': r.index, 'status': r.status,
                     'evaluation_requests': r.evaluation_requests,
                     'starter_selection': r.starter_selection,
                     'ls_preparation': None if r.ls_preparation is None else
                         {k: v for k, v in r.ls_preparation.items() if k != 'soft_quench'},
                     'ls_update': r.ls_update} for r in result.records],
        'pool_decisions': copy.deepcopy(adapter.decisions),
    }


def fresh_terminal(folder, result, atoms, item, config, budget, label):
    calc = make_calculator(item)
    surface = SharedSurface(calc, folder / 'fresh.jsonl', budget, label, kind='fresh')
    energies = [float(minimum.energy) for minimum in result.minima]
    expected = {'current': getattr(result.checkpoint, 'current_energy', None),
                'best': min(energies) if energies else None}
    checks = []
    for candidate_label, candidate in (('current', result.current), ('best', result.best)):
        try:
            energy, forces = surface.evaluate(candidate)
            error = (None if expected[candidate_label] is None else
                     energy - float(expected[candidate_label]))
            fmax = float(np.linalg.norm(forces, axis=1).max())
            checks.append({'label': candidate_label, 'energy_eV': energy,
                           'expected_energy_eV': expected[candidate_label],
                           'energy_error_eV': error,
                           'force_threshold_eV_A': float(config.fmax),
                           'force_qualified': bool(fmax <= config.fmax),
                           'fmax_eV_A': fmax,
                           'composition_unchanged': bool(np.array_equal(candidate.numbers, atoms.numbers)),
                           'cell_unchanged': bool(np.array_equal(candidate.cell.array, atoms.cell.array)),
                           'pbc_unchanged': bool(np.array_equal(candidate.pbc, atoms.pbc))})
        except Exception as error:
            checks.append({'label': candidate_label, 'status': 'error', 'error': repr(error)})
    return checks


def run_case(out, plan, name):
    item, atoms, payload, config = make_case(plan, name)
    budget = SharedBudget(plan['budgets']['search_requests_per_case_all_legs'],
                          plan['budgets']['fresh_terminal_checks_per_case_all_legs'],
                          plan['budgets']['wall_seconds_per_case'])
    case_out = out / name; case_out.mkdir()
    ledger = case_out / 'requests.jsonl'
    rows = []
    from pamssw.standalone import NativeMCSettings, load_ssw_checkpoint, run_ssw
    from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter
    def make_mc():
        return (NativeMCSettings(energy_tol=float(payload['native_mc']['energy_tol_eV']),
                                 maxtrap=int(payload['native_mc']['maxtrap']))
                if 'native_mc' in payload else None)
    adapter = PoolStarterAdapter(mode='pam', energy_tol=plan['pool']['energy_tol'],
                                 rmsd_tol=plan['pool']['rmsd_tol'])
    selector_rng = np.random.default_rng(item['seed'] + 1000003)
    continuous_kernel_rng = np.random.default_rng(item['seed'])
    continuous_result = None
    resumed_result = None
    try:
        (case_out / 'continuous2').mkdir()
        continuous_surface = SharedSurface(make_calculator(item), ledger, budget, 'continuous2')
        result = run_ssw(atoms.copy(), continuous_surface, steps=2, config=config,
            rng=continuous_kernel_rng, ls=prior.make_ls(item, payload), mc=make_mc(),
            starter_selector=adapter, selector_rng=selector_rng,
            checkpoint_path=case_out / 'continuous-checkpoint.pkl')
        continuous_result = result
        rows.append({'leg': 'continuous2', **record_result(case_out / 'continuous2', result,
                     adapter, selector_rng, continuous_kernel_rng, continuous_surface),
                     'fresh_checks': fresh_terminal(case_out / 'continuous2', result, atoms,
                                                    item, config, budget, 'continuous2')})
    except Exception as error:
        rows.append({'leg': 'continuous2', 'status': 'exception', 'error': repr(error)})
    # Split arm uses fresh adapter/calculator on resume; no finalize occurs on any leg.
    first_dir, resume_dir = case_out / 'first1', case_out / 'resume1'
    first_dir.mkdir(); resume_dir.mkdir()
    first_adapter = PoolStarterAdapter(mode='pam', energy_tol=plan['pool']['energy_tol'],
                                       rmsd_tol=plan['pool']['rmsd_tol'])
    first_rng = np.random.default_rng(item['seed'] + 1000003)
    first_kernel_rng = np.random.default_rng(item['seed'])
    checkpoint_path = first_dir / 'checkpoint.pkl'
    try:
        first_surface = SharedSurface(make_calculator(item), ledger, budget, 'first1')
        first = run_ssw(atoms.copy(), first_surface, steps=1, config=config,
                        rng=first_kernel_rng, ls=prior.make_ls(item, payload), mc=make_mc(),
                        starter_selector=first_adapter, selector_rng=first_rng,
                        checkpoint_path=checkpoint_path)
        first_summary = record_result(first_dir, first, first_adapter, first_rng,
                                      first_kernel_rng, first_surface)
        rows.append({'leg': 'first1', **first_summary})
        checkpoint = load_ssw_checkpoint(checkpoint_path)
        resumed_adapter = PoolStarterAdapter(mode='pam', energy_tol=plan['pool']['energy_tol'],
                                             rmsd_tol=plan['pool']['rmsd_tol'])
        resumed_rng = np.random.default_rng(item['seed'] + 1000003)
        resumed_kernel_rng = np.random.default_rng(item['seed'])
        resume_surface = SharedSurface(make_calculator(item), ledger, budget, 'resume1')
        resumed = run_ssw(checkpoint.current, resume_surface, steps=1, config=config,
                          rng=resumed_kernel_rng, ls=prior.make_ls(item, payload), mc=make_mc(),
                          starter_selector=resumed_adapter, selector_rng=resumed_rng,
                          checkpoint=checkpoint, checkpoint_path=resume_dir / 'checkpoint.pkl')
        resumed_result = resumed
        rows.append({'leg': 'resume1', **record_result(resume_dir, resumed, resumed_adapter,
                     resumed_rng, resumed_kernel_rng, resume_surface),
                     'fresh_checks': fresh_terminal(resume_dir, resumed, atoms, item, config,
                                                    budget, 'resume1')})
        rows[-1]['first_pool_export'] = first_adapter.export_state()
        if continuous_result is not None:
            rows[-1]['position_max_abs_diff_vs_continuous'] = float(
                np.max(np.abs(continuous_result.current.positions - resumed.current.positions)))
            rows[-1]['cell_max_abs_diff_vs_continuous'] = float(
                np.max(np.abs(continuous_result.current.cell.array - resumed.current.cell.array)))
            rows[-1]['pbc_equal_vs_continuous'] = bool(
                np.array_equal(continuous_result.current.pbc, resumed.current.pbc))
            rows[-1]['composition_equal_vs_continuous'] = bool(
                np.array_equal(continuous_result.current.numbers, resumed.current.numbers))
    except Exception as error:
        rows.append({'leg': 'resume1' if rows and rows[-1].get('leg') == 'first1'
                     else 'first1', 'status': 'exception', 'error': repr(error)})
    dump(case_out / 'summary.json', {'case': name, 'rows': rows,
        'search_requests': budget.search_requests, 'fresh_requests': budget.fresh_requests,
        'total_ef': budget.search_requests + budget.fresh_requests,
        'wall_seconds': time.monotonic() - budget.started,
        'within_budget': budget.search_requests <= plan['budgets']['search_requests_per_case_all_legs'] and
                         budget.fresh_requests <= plan['budgets']['fresh_terminal_checks_per_case_all_legs']})


def validate_only(plan):
    """Parse every prepared input/configuration without importing MACE or evaluating a PES."""
    budgets = plan['budgets']
    required = ('search_requests_per_case_all_legs',
                'fresh_terminal_checks_per_case_all_legs',
                'total_ef_all_cases', 'wall_seconds_per_case')
    missing = [key for key in required if key not in budgets]
    if missing:
        raise ValueError(f'missing budget keys: {missing}')
    if len(plan['cases']) * (budgets['search_requests_per_case_all_legs'] +
                             budgets['fresh_terminal_checks_per_case_all_legs']) != budgets['total_ef_all_cases']:
        raise ValueError('total_ef_all_cases does not match per-case caps')
    from pamssw.standalone import NativeMCSettings
    checked = []
    for name in plan['cases']:
        item, atoms, payload, config = make_case(plan, name)
        ls = prior.make_ls(item, payload)
        mc = None
        if 'native_mc' in payload:
            native = payload['native_mc']
            mc = NativeMCSettings(energy_tol=float(native['energy_tol_eV']),
                                  maxtrap=int(native['maxtrap']))
        checked.append({'case': name, 'atoms': len(atoms), 'seed': item['seed'],
                        'model': item['model'], 'ls': jsonable(ls),
                        'native_mc': None if mc is None else jsonable(mc),
                        'fmax_eV_A': float(config.fmax)})
    print(json.dumps({'status': 'VALIDATED_ONLY', 'cases': checked,
                      'budgets': budgets}, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads((HERE / 'protocol.json').read_text())
    if args.validate_only:
        validate_only(plan)
        return
    if args.output is None:
        parser.error('--output is required with --execute')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    dump(out / 'protocol.json', plan)
    launch_provenance(out)
    for name in plan['cases']:
        try:
            run_case(out, plan, name)
        except Exception as error:
            # Preserve one case's protocol/runtime failure and still qualify the other.
            case_dir = out / name
            case_dir.mkdir(parents=True, exist_ok=True)
            dump(case_dir / 'summary.json', {'case': name, 'status': 'exception',
                                             'error': repr(error)})


if __name__ == '__main__': main()
