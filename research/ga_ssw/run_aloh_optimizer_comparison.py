"""Bounded AlOH optimizer audit with frozen-source execution.

Prepare copies the complete PAM-SSW tree. Execute starts that copied runner;
the child imports only that copy. Paid records and failure/censor summaries
are retained. This research runner does not modify PAM-SSW production code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE1-AlOH/addition/add.arc')
MODEL = Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
OPTIMIZERS = ('safe-lbfgs-total', 'ase-lbfgs-linesearch', 'scipy-lbfgsb')
GLOBAL_WALL_SECONDS = 400.0


class ArmTimeout(RuntimeError):
    """Alarm exception intentionally caught by run_ssw's runtime-error path."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def enc(value):
    try:
        from ase import Atoms
        if isinstance(value, Atoms):
            return {'symbols': value.get_chemical_symbols(), 'numbers': value.numbers.tolist(),
                    'positions': value.positions.tolist(), 'cell': value.cell.array.tolist(),
                    'pbc': value.pbc.tolist()}
    except ImportError:
        pass
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except Exception:
            pass
    from dataclasses import fields, is_dataclass
    if is_dataclass(value):
        return {field.name: enc(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): enc(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [enc(item) for item in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path, value):
    path.write_text(json.dumps(enc(value), indent=2, allow_nan=False, default=str) + '\n')


def append_jsonl(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(enc(value), allow_nan=False, default=str) + '\n')


def prepare(out):
    from ase import __version__ as ase_version
    from ase.io import read

    out.mkdir(parents=True, exist_ok=True)
    source = out / 'source'
    shutil.copytree(ROOT / 'pamssw', source / 'pamssw', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, out / 'runner.py')
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    shutil.copy2(INPUT, inputs / 'aloh.add.arc')
    frames = []
    for index in (0, 1):
        atoms = read(INPUT, index=index)
        frames.append({'index': index, 'numbers': atoms.numbers.tolist(),
                       'positions': atoms.positions.tolist(), 'cell': atoms.cell.array.tolist(),
                       'pbc': atoms.pbc.tolist(), 'formula': atoms.get_chemical_formula()})
    plan = {
        'status': 'prepared', 'input_source': str(INPUT), 'input_sha256': sha256(INPUT),
        'model': str(MODEL), 'model_sha256': sha256(MODEL), 'frames': frames,
        'seed': 11, 'arms': list(OPTIMIZERS),
        'config': {'width': .6, 'rotation_bias': 100., 'max_gaussians': 10,
                   'temperature_K': 400., 'fmax': .03, 'relax_steps': 300,
                   'fd_step': 1e-4, 'rotation_hvp': 100, 'rotation_tol': .02,
                   'forward_force': .1, 'direction_sampling': 'global',
                   'rotation_solver': 'dimer', 'cluster_frame': 'translation_only',
                   'bias_fmax': .1},
        'budgets': {'search_requests': 600, 'fresh_requests': 11,
                    'arm_wall_seconds': 90., 'global_wall_seconds': GLOBAL_WALL_SECONDS},
        'backend': {'requested': 'MACECalculator CPU float64', 'model': str(MODEL)},
        'software': {'python': sys.version, 'ase': ase_version},
    }
    write_json(out / 'plan.json', plan)
    write_json(out / 'runtime-import.json', {'expected_source': str(source / 'pamssw'),
                                               'runner_source': str(out / 'runner.py')})
    print(json.dumps({'prepared': str(out), 'frames': [(x['index'], x['formula']) for x in frames]}))


def make_calculator(plan, backend):
    if backend == 'tblite':
        from tblite.ase import TBLite
        return TBLite(method='GFN2-xTB', accuracy=.001, verbosity=0)
    if backend == 'emt':
        from ase.calculators.emt import EMT
        return EMT()
    from mace.calculators import MACECalculator
    head = plan.get('backend', {}).get('head')
    kwargs = {} if head is None else {'head': head}
    calculator = MACECalculator(model_paths=plan['model'],
                          device=plan.get('backend', {}).get('device', 'cpu'),
                          default_dtype='float64', enable_cueq=False, enable_oeq=False,
                          **kwargs)
    if head is not None and calculator.head != head:
        raise ValueError(f'requested MACE head {head!r}, resolved {calculator.head!r}')
    return calculator


def run_arm(arm_dir, frame, optimizer, plan, out, backend, global_deadline):
    import numpy as np
    from ase.io import read
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw

    arm_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = arm_dir / 'ledger.jsonl'
    started = time.monotonic()
    arm_deadline = min(global_deadline, started + float(plan['budgets']['arm_wall_seconds']))
    search_deadline = arm_deadline - float(plan['budgets'].get('fresh_reserve_seconds', 0.))
    search_started = None
    search_finished = None
    fresh_started = None
    result = None
    error = None
    timed_out = False
    censor_reason = None
    denials = 0
    search_cap = int(plan['budgets']['search_requests'])
    fresh_cap = int(plan['budgets']['fresh_requests'])
    atoms = read(out / plan.get('input_file', 'inputs/aloh.add.arc'), index=frame['index'])
    ls = None
    if plan.get('native_ls') is not None:
        from pamssw.standalone import NativeLSSettings, LSPrequenchSettings
        from pamssw.standalone.native_ls import HCO_BOND_ENERGIES, HCO_BOND_LENGTHS
        settings = plan['native_ls']
        ls = NativeLSSettings(HCO_BOND_ENERGIES, HCO_BOND_LENGTHS,
            target_mev_per_atom=settings['target_mev_per_atom'],
            prequench=LSPrequenchSettings(**settings['prequench']))
    surface = ASESurface(make_calculator(plan, backend))
    original_evaluate = surface.evaluate

    def evaluate(candidate):
        nonlocal denials, timed_out, censor_reason
        before = surface.requests
        if before >= search_cap or time.monotonic() >= search_deadline:
            denials += 1
            censor_reason = 'request_cap' if before >= search_cap else 'search_wall_cap'
            if time.monotonic() >= search_deadline:
                timed_out = True
            append_jsonl(ledger_path, {'kind': 'denial',
                        'reason': censor_reason,
                        'before': before, 'after': before, 'charged': False,
                        'positions': candidate.positions, 'cell': candidate.cell.array,
                        'pbc': candidate.pbc})
            raise ArmTimeout('search request or arm wall limit exhausted')
        try:
            energy, forces = original_evaluate(candidate)
        except Exception as exc:
            append_jsonl(ledger_path, {'kind': 'failure', 'before': before,
                        'after': surface.requests, 'charged': surface.requests > before,
                        'positions': candidate.positions, 'cell': candidate.cell.array,
                        'pbc': candidate.pbc, 'error': f'{type(exc).__name__}: {exc}'})
            raise
        append_jsonl(ledger_path, {'kind': 'evaluation', 'before': before,
                    'after': surface.requests, 'charged': surface.requests > before,
                    'energy': float(energy), 'fmax': float(np.linalg.norm(forces, axis=1).max()),
                    'positions': candidate.positions, 'cell': candidate.cell.array,
                    'pbc': candidate.pbc})
        return energy, forces

    surface.evaluate = evaluate
    cfg = SSWConfig(**plan['config'], quench_optimizer=optimizer)
    write_json(arm_dir / 'resolved-settings.json', {'config': cfg, 'ls': ls})
    def alarm(_signum, _frame):
        nonlocal timed_out, censor_reason
        timed_out = True
        censor_reason = 'search_wall_cap' if fresh_started is None else 'arm_wall_cap'
        raise ArmTimeout('arm/global wall limit')

    old_handler = signal.signal(signal.SIGALRM, alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, search_deadline - time.monotonic()))
        search_started = time.monotonic()
        try:
            result = run_ssw(atoms, surface, steps=int(plan.get('outer_steps', 10)), config=cfg,
                             rng=np.random.default_rng(plan['seed']),
                             ls=ls,
                             checkpoint_path=arm_dir / 'boundary.pkl')
        except ArmTimeout:
            error = traceback.format_exc()
        if timed_out:
            error = error or 'ArmTimeout: arm/global wall limit'
        search_finished = time.monotonic()
    except Exception:
        error = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        search_finished = time.monotonic()

    fresh = []
    if result is not None:
        write_json(arm_dir / 'result.json', result)
        import pickle
        with (arm_dir / 'result.pkl').open('wb') as stream:
            pickle.dump(result, stream)
        fresh_started = time.monotonic()
        minima = list(result.minima[:fresh_cap])
        try:
            if time.monotonic() < arm_deadline:
                signal.setitimer(signal.ITIMER_REAL, max(.001, arm_deadline - time.monotonic()))
                fresh_surface = ASESurface(make_calculator(plan, backend))
                for index, minimum in enumerate(minima):
                    item = {'index': index, 'status': 'failed'}
                    if fresh_surface.requests >= fresh_cap or time.monotonic() >= arm_deadline:
                        item.update(status='censored', reason='fresh_cap_or_arm_wall')
                    else:
                        try:
                            calculator = fresh_surface.calculator
                            if hasattr(calculator, 'reset'):
                                calculator.reset()
                            energy, forces = fresh_surface.evaluate(minimum.atoms)
                            fmax = float(np.linalg.norm(forces, axis=1).max())
                            item.update(status='checked', energy=float(energy), fmax=fmax,
                                        requests=fresh_surface.requests,
                                        energy_error=float(energy - minimum.energy),
                                        numbers_exact=bool(np.array_equal(minimum.atoms.numbers, atoms.numbers)),
                                        cell_exact=bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array)),
                                        pbc_exact=bool(np.array_equal(minimum.atoms.pbc, atoms.pbc)),
                                        qualified=bool(np.isfinite(energy) and np.isfinite(fmax)
                                            and abs(energy - minimum.energy) <= 1e-7
                                            and fmax <= cfg.fmax
                                            and np.array_equal(minimum.atoms.numbers, atoms.numbers)
                                            and np.array_equal(minimum.atoms.cell.array, atoms.cell.array)
                                            and np.array_equal(minimum.atoms.pbc, atoms.pbc)))
                        except ArmTimeout:
                            timed_out = True
                            item.update(status='censored', reason='arm_wall_cap')
                            fresh.append(item)
                            break
                        except Exception as exc:
                            item['error'] = f'{type(exc).__name__}: {exc}'
                    fresh.append(item)
            else:
                timed_out = True
                fresh = [{'index': index, 'status': 'censored', 'reason': 'arm_wall_cap'}
                         for index in range(len(minima))]
        except ArmTimeout:
            timed_out = True
            fresh.extend({'index': index, 'status': 'censored', 'reason': 'arm_wall_cap'}
                         for index in range(len(fresh), len(minima)))
        except Exception as exc:
            fresh = [{'index': index, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}' }
                     for index in range(len(minima))]
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old_handler)
    write_json(arm_dir / 'fresh.json', fresh)
    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line] if ledger_path.exists() else []
    summary = {'frame': frame['index'], 'optimizer': optimizer, 'backend': backend,
               'status': 'censored' if timed_out or censor_reason else (result.status if result is not None else 'failed'),
               'censor_reason': censor_reason or ('arm_wall_cap' if timed_out else None),
               'error': error, 'timed_out': timed_out,
               'search_requests': sum(int(item.get('charged', False)) for item in events),
               'ledger_events': len(events), 'denials': denials,
               'elapsed_seconds': time.monotonic() - started,
               'search_wall_seconds': search_finished - search_started if search_started and search_finished else None,
               'fresh_wall_seconds': time.monotonic() - fresh_started if fresh_started else 0.,
               'result_present': result is not None,
               'records': len(result.records) if result is not None else None,
               'evaluation_requests': result.evaluation_requests if result is not None else None,
               'minima': len(result.minima) if result is not None else None,
               'fresh_checked': sum(item.get('status') == 'checked' for item in fresh),
               'fresh_failures': sum(item.get('status') == 'failed' for item in fresh),
               'search_cap': search_cap, 'fresh_cap': fresh_cap}
    write_json(arm_dir / 'summary.json', summary)
    if error:
        (arm_dir / 'exception.txt').write_text(error)
    return summary


def child(out, backend):
    sys.path.insert(0, str(out / 'source'))
    import pamssw
    plan = json.loads((out / 'plan.json').read_text())
    imported = Path(pamssw.__file__).resolve()
    source = (out / 'source' / 'pamssw').resolve()
    if not imported.is_relative_to(source):
        raise RuntimeError(f'non-frozen import: {imported}')
    write_json(out / 'runtime-import.json', {'pamssw_file': str(imported),
                                               'runner_file': str(Path(__file__).resolve()),
                                               'source_root': str(source), 'backend': backend})
    global_deadline = time.monotonic() + float(plan['budgets']['global_wall_seconds'])
    rows = []
    for frame in plan['frames']:
        for optimizer in plan['arms']:
            arm_dir = out / f"frame{frame['index']}-{optimizer}"
            if time.monotonic() >= global_deadline:
                arm_dir.mkdir(parents=True, exist_ok=True)
                summary = {'frame': frame['index'], 'optimizer': optimizer, 'backend': backend,
                           'status': 'not_run', 'reason': 'global_wall_cap',
                           'search_requests': 0, 'fresh_checked': 0, 'result_present': False,
                           'search_cap': plan['budgets']['search_requests'],
                           'fresh_cap': plan['budgets']['fresh_requests']}
                write_json(arm_dir / 'summary.json', summary)
            else:
                try:
                    summary = run_arm(arm_dir, frame, optimizer, plan, out, backend, global_deadline)
                except BaseException as exc:
                    # Setup/model failures are configuration outcomes, not reasons
                    # to lose the arm denominator or any ledger already written.
                    arm_dir.mkdir(parents=True, exist_ok=True)
                    ledger_path = arm_dir / 'ledger.jsonl'
                    events = ([json.loads(line) for line in ledger_path.read_text().splitlines() if line]
                              if ledger_path.exists() else [])
                    summary = {'frame': frame['index'], 'optimizer': optimizer,
                               'backend': backend, 'status': 'failed',
                               'error': f'{type(exc).__name__}: {exc}',
                               'search_requests': sum(int(item.get('charged', False)) for item in events),
                               'ledger_events': len(events), 'fresh_checked': 0,
                               'result_present': False}
                    (arm_dir / 'exception.txt').write_text(traceback.format_exc())
                    write_json(arm_dir / 'summary.json', summary)
            rows.append(summary)
            write_json(out / 'summary.json', rows)
            print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--child', type=Path)
    parser.add_argument('--backend', choices=('mace', 'emt', 'tblite'), default='mace')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.execute:
        runner = args.execute.resolve() / 'runner.py'
        if not runner.is_file():
            raise FileNotFoundError(f'frozen runner missing: {runner}')
        env = os.environ.copy()
        env.update({'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        raise SystemExit(subprocess.call([sys.executable, str(runner), '--child',
                                          str(args.execute.resolve()), '--backend', args.backend], env=env))
    elif args.child:
        child(args.child.resolve(), args.backend)
    else:
        parser.error('choose --prepare or --execute')


if __name__ == '__main__':
    main()
