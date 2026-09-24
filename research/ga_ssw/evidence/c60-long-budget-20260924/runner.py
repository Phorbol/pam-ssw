#!/usr/bin/env python3
"""One bounded segment of a frozen C60 SSW/LS experiment (research harness).

No automatic submission, retry, input replacement, or budget redistribution.
Every attempted E/F request is durably charged before invoking the calculator.
Allocation reservations include startup and remain charged after interruption.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class Budget:
    """Durable caller costs, independent of the resumable algorithm lineage."""
    def __init__(self, folder, plan):
        self.folder, self.plan = Path(folder), plan
        self.path = self.folder / 'budget.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'plan_sha256': sha(self.folder / 'plan.json'), 'search': 0, 'fresh': 0,
            'reserved_seconds': 0, 'segments': [], 'status': 'ready', 'fresh_checks': {}}
        if self.state['plan_sha256'] != sha(self.folder / 'plan.json'):
            raise ValueError('frozen plan changed')
        self.started = None
        self.deadline = None
        self.save()

    def save(self):
        atomic_json(self.path, self.state)

    def begin(self, seconds, *, interrupted=False):
        if self.state['status'] not in ('ready', 'paused'):
            if not (interrupted and self.state['status'] == 'running'):
                raise ValueError('trajectory is terminal or interrupted; inspect before explicit recovery')
        if seconds <= 0 or self.state['reserved_seconds'] + seconds > self.plan['wall_seconds']:
            raise ValueError('cumulative allocation budget exceeded')
        if self.state['search'] >= self.plan['search_cap']:
            raise ValueError('search request budget exhausted')
        self.state['reserved_seconds'] += seconds
        self.state['segments'].append({'reserved_seconds': seconds, 'job_id': os.getenv('SLURM_JOB_ID'),
                                       'started_unix': time.time(), 'state': 'running'})
        self.state['status'] = 'running'
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.save()

    def charge(self, category):
        if self.state[category] >= self.plan[category + '_cap']:
            raise RuntimeError(category + '_budget_exhausted')
        if self.deadline is not None and time.monotonic() >= self.deadline - 60:
            raise RuntimeError('segment_hard_deadline')
        self.state[category] += 1
        self.save()  # precedes even a failing calculator call; never refunded
        return self.state[category]

    def finish(self, status, **details):
        self.state['status'] = status
        self.state['segments'][-1].update(state=status,
            elapsed_seconds=time.monotonic() - self.started, **details)
        self.save()


def calculator(plan):
    if plan.get('backend') == 'emt':
        from ase.calculators.emt import EMT
        return EMT()
    if sha(plan['model']) != plan['model_sha256']:
        raise ValueError('model changed')
    import torch
    from mace.calculators import MACECalculator
    torch.set_num_threads(1); torch.manual_seed(plan['runtime']['torch_manual_seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return MACECalculator(model_paths=plan['model'], head=plan['head'],
        device=plan['device'], default_dtype=plan['dtype'], enable_cueq=False, enable_oeq=False)


def settings(plan):
    from pamssw.standalone import LSPrequenchSettings, NativeLSSettings
    spec = plan.get('native_ls')
    if spec is None:
        return None
    spec = dict(spec); spec.pop('parameter_source', None)
    for key in ('bond_energies', 'bond_lengths'):
        spec[key] = {tuple(map(int, k.split(','))): v for k, v in spec[key].items()}
    spec['prequench'] = LSPrequenchSettings(**spec['prequench'])
    return NativeLSSettings(**spec)


def run_segment(folder, seconds, *, max_attempts=None, interrupted=False):
    import numpy as np
    from ase.io import read
    from pamssw.standalone import (SSWConfig, NativeMCSettings, RecoveredRotationSettings,
        run_ssw, load_ssw_checkpoint, save_ssw_checkpoint)
    from pamssw.standalone.surface import ASESurface
    folder = Path(folder); plan = json.loads((folder / 'plan.json').read_text())
    if sha(folder / plan['input']) != plan['input_sha256']:
        raise ValueError('input changed')
    for filename, expected in plan.get('frozen_files', {}).items():
        if sha(folder / filename) != expected:
            raise ValueError('frozen file changed: ' + filename)
    budget = Budget(folder, plan)
    cp_path = folder / 'checkpoint.pkl'
    cp = load_ssw_checkpoint(cp_path) if cp_path.exists() else None
    if cp is not None and cp.status != 'completed':
        raise ValueError('terminal checkpoint cannot resume')
    if cp is not None and cp.evaluation_requests > budget.state['search']:
        raise ValueError('checkpoint exceeds paid request ledger')
    if interrupted and cp is None:
        raise ValueError('no safe checkpoint; cannot repeat initial quench')
    budget.begin(seconds, interrupted=interrupted)
    calc = calculator(plan)
    calc_calls = 0
    original_calculate = calc.calculate
    def counted_calculate(*args, **kwargs):
        nonlocal calc_calls
        calc_calls += 1
        return original_calculate(*args, **kwargs)
    calc.calculate = counted_calculate
    raw = ASESurface(calc)
    trace = (folder / 'requests.jsonl').open('a', buffering=1)
    class PaidSurface:
        requests = 0
        def evaluate(self, atoms):
            request_id = budget.charge('search')
            self.requests += 1
            start = time.monotonic()
            row = {'id': request_id, 'segment': len(budget.state['segments'])}
            try:
                energy, force = raw.evaluate(atoms)
                row.update(energy_eV=energy, fmax_eV_A=float(np.linalg.norm(force, axis=1).max()))
                return energy, force
            except Exception as error:
                row['error'] = repr(error)
                raise
            finally:
                row['elapsed_seconds'] = time.monotonic() - start
                trace.write(json.dumps(row, allow_nan=False) + '\n')
    surface = PaidSurface()
    start_index = cp.next_index if cp else 0
    safe_count = 0
    def boundary(snapshot):
        nonlocal safe_count
        save_ssw_checkpoint(cp_path, snapshot)
        safe_count += 1
        atomic_json(folder / 'boundary.json', {
            'next_index': snapshot.next_index, 'lineage_requests': snapshot.evaluation_requests,
            'paid_requests': budget.state['search'], 'segment': len(budget.state['segments']),
            'checkpoint_bytes': cp_path.stat().st_size, 'best_energy_eV': snapshot.best.energy})
        # Leave allocation time for finishing the checkpoint and final qualification.
        return (time.monotonic() >= budget.deadline - min(600, seconds / 4)
                or (max_attempts is not None and safe_count >= max_attempts)
                or budget.state['search'] >= plan['search_cap'])
    try:
        kwargs = dict(config=SSWConfig(**plan['ssw_config']),
            rng=np.random.default_rng(plan['seed']), checkpoint=cp,
            checkpoint_callback=boundary, ls=settings(plan))
        if plan.get('native_mc') is not None:
            kwargs['mc'] = NativeMCSettings(**plan['native_mc'])
        if plan.get('recovered_rotation') is not None:
            kwargs['recovered_rotation'] = RecoveredRotationSettings(**plan['recovered_rotation'])
        result = run_ssw(read(folder / plan['input']), surface,
            steps=plan['search_cap'], **kwargs)  # cap cannot restrict before request budget
        if result.status == 'paused':
            status = 'paused' if budget.state['search'] < plan['search_cap'] else 'search_exhausted'
        else:
            status = 'search_exhausted' if budget.state['search'] >= plan['search_cap'] else result.status
        if result.checkpoint is not None and result.checkpoint.status != 'completed':
            save_ssw_checkpoint(folder / 'terminal.pkl', result.checkpoint)
        # End-of-budget best/landings may be newer than the last safe checkpoint.
        if result.checkpoint is not None:
            save_ssw_checkpoint(folder / 'last-result.pkl', result.checkpoint)
        budget.finish(status, algorithm_status=result.status, start_index=start_index,
            safe_boundaries=safe_count, segment_search_requests=surface.requests,
            calculate_calls=calc_calls)
    except Exception as error:
        budget.finish('failed', error=repr(error), segment_search_requests=surface.requests,
                      calculate_calls=calc_calls)
        raise
    finally:
        trace.close()
    if budget.state['reserved_seconds'] == plan['wall_seconds'] and budget.state['status'] == 'paused':
        budget.state['status'] = 'wall_exhausted'; budget.save()
    return budget.state


def finalize(folder):
    import numpy as np
    from ase.io import read, write
    from pamssw.standalone import load_ssw_checkpoint
    from pamssw.standalone.surface import ASESurface
    folder = Path(folder); plan = json.loads((folder / 'plan.json').read_text())
    budget = Budget(folder, plan)
    if budget.state['status'] not in ('search_exhausted', 'wall_exhausted', 'failed', 'completed'):
        raise ValueError('finalize only terminal trajectory')
    path = folder / 'last-result.pkl'
    if not path.exists():
        path = folder / 'checkpoint.pkl'
    if not path.exists():
        atomic_json(folder / 'summary.json', {'status': budget.state['status'], 'no_minima': True})
        return
    cp = load_ssw_checkpoint(path)
    original = read(folder / plan['input'])
    sys.path.insert(0, str(folder))
    from validator import graph_row
    candidates = [('initial', cp.initial), ('best', cp.best)]
    graphs = []
    first_cage = None
    for index, minimum in enumerate(cp.minima):
        row = graph_row(minimum.atoms.numbers, minimum.atoms.positions, 1.8)
        graphs.append({'index': index, 'energy_eV': minimum.energy, **row})
        if first_cage is None and minimum.converged and row['graph_cage_candidate']:
            first_cage = minimum
    if first_cage is not None and all(not np.array_equal(first_cage.atoms.positions, m.atoms.positions)
                                      for _, m in candidates):
        candidates.append(('first_cage', first_cage))
    calc = calculator(plan)
    for label, minimum in candidates:
        if label in budget.state['fresh_checks']:
            continue  # includes interrupted fresh attempts; never spend twice
        budget.state['fresh_checks'][label] = {'status': 'charged_unfinished'}
        budget.charge('fresh')
        row = {'status': 'failed'}
        try:
            calc.reset(); energy, force = ASESurface(calc).evaluate(minimum.atoms)
            fmax = float(np.linalg.norm(force, axis=1).max())
            geometry = all(np.array_equal(getattr(minimum.atoms, k), getattr(original, k))
                           for k in ('numbers', 'cell', 'pbc'))
            qualified = bool(minimum.converged and fmax <= plan['ssw_config']['fmax']
                             and abs(energy - minimum.energy) <= 1e-6 and geometry)
            checks = [graph_row(minimum.atoms.numbers, minimum.atoms.positions, cut)
                      for cut in (1.8, 1.64, 1.7)]
            row = dict(status='completed', energy_eV=energy, fmax_eV_A=fmax,
                numerical_qualified=qualified, graphs=checks,
                energy_target=bool(qualified and energy <= plan['reference_energy_eV'] + .01),
                joint_target=bool(qualified and checks[0]['graph_cage_candidate']
                                  and energy <= plan['reference_energy_eV'] + .01))
            write(folder / (label + '-fresh.traj'), minimum.atoms)
        except Exception as error:
            row['error'] = repr(error)
        budget.state['fresh_checks'][label] = row; budget.save()
    atomic_json(folder / 'summary.json', dict(budget.state, graphs=graphs,
        lineage_requests=cp.evaluation_requests, saved_outer_attempts=cp.next_index,
        scope='two-input fixed-protocol comparison on MH1, not a general success-rate estimate'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--segment-seconds', type=int)
    parser.add_argument('--max-attempts', type=int)
    parser.add_argument('--resume-interrupted', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required')
    with (args.run_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.finalize:
            finalize(args.run_dir)
        else:
            if not args.segment_seconds:
                parser.error('--segment-seconds is required')
            state = run_segment(args.run_dir, args.segment_seconds,
                max_attempts=args.max_attempts, interrupted=args.resume_interrupted)
            if state['status'] in ('search_exhausted', 'wall_exhausted', 'completed', 'failed'):
                finalize(args.run_dir)

if __name__ == '__main__':
    main()
