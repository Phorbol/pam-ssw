#!/usr/bin/env python3
"""One bounded segment of a frozen C60 SSW/LS experiment (research harness).

No automatic submission, retry, input replacement, or budget redistribution.
A durable 64-request reservation precedes E/F calls; clean boundaries reconcile
actual charges, while hard interruptions retain the uncertain reservation.
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
        # Only an unfinished process can leave an uncertain issued-request count.
        # Consume its durable reservation conservatively rather than refund work.
        reserved = self.state.get('search_reserved', self.state['search'])
        if self.state['status'] == 'running' and reserved > self.state['search']:
            self.state['unconfirmed_search_reservations'] = (
                self.state.get('unconfirmed_search_reservations', 0) + reserved - self.state['search'])
            self.state['search'] = reserved
        self.state['search_reserved'] = self.state['search']
        self.io_seconds = 0.
        self.started = None
        self.deadline = None
        self.save()

    def save(self):
        started = time.monotonic()
        atomic_json(self.path, self.state)
        self.io_seconds += time.monotonic() - started

    def sync_search(self):
        # Called only when no E/F request is in flight. Unused quota is not work.
        self.state['search_reserved'] = self.state['search']
        self.save()

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
        self.io_seconds = 0.  # segment metric excludes constructor I/O before timing
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.save()

    def charge(self, category):
        if self.state[category] >= self.plan[category + '_cap']:
            raise RuntimeError(category + '_budget_exhausted')
        if self.deadline is not None and time.monotonic() >= self.deadline - 60:
            raise RuntimeError('segment_hard_deadline')
        if category == 'search':
            if self.state['search'] >= self.state['search_reserved']:
                # Fixed engineering granularity: at most 64 uncertain requests
                # after a hard interruption, independent of the search algorithm.
                self.state['search_reserved'] = min(self.plan['search_cap'], self.state['search'] + 64)
                self.save()  # durable upper bound BEFORE any request in this block
            self.state['search'] += 1
        else:
            self.state[category] += 1
            self.save()
        return self.state[category]

    def finish(self, status, **details):
        self.state['status'] = status
        self.state['segments'][-1].update(state=status,
            elapsed_seconds=time.monotonic() - self.started, budget_io_seconds=self.io_seconds, **details)
        self.sync_search()


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
    from importlib import metadata
    versions = {}
    for package in ('numpy', 'scipy', 'ase', 'mace-torch', 'torch'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    atomic_json(folder / f"environment-{len(budget.state['segments'])}.json", {
        'python': sys.version, 'executable': sys.executable, 'packages': versions,
        'node': os.uname().nodename, 'job_id': os.getenv('SLURM_JOB_ID'),
        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES')})
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
        budget.sync_search()
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
            kwargs['mc'] = NativeMCSettings(plan['native_mc']['energy_tol_eV'],
                                            plan['native_mc']['maxtrap'])
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
        failed_initial = getattr(error, 'result', None)
        if failed_initial is not None:
            from ase.io import write
            write(folder / 'failed-initial.traj', failed_initial.atoms)
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
    selected_cage = None
    selected_index = None
    first_joint_index = None
    for index, minimum in enumerate(cp.minima):
        row = graph_row(minimum.atoms.numbers, minimum.atoms.positions, 1.8)
        graphs.append({'index': index, 'energy_eV': minimum.energy, **row})
        if minimum.converged and row['graph_cage_candidate']:
            if minimum.energy <= plan['reference_energy_eV'] + .01 and first_joint_index is None:
                first_joint_index = index
            if selected_cage is None or minimum.energy < selected_cage.energy:
                selected_cage, selected_index = minimum, index
    # Preserve the first stored joint hit for independent qualification; otherwise
    # qualify the lowest-energy cage. A later non-cage best must not hide it.
    if first_joint_index is not None:
        selected_index = first_joint_index
        selected_cage = cp.minima[selected_index]
    if selected_cage is not None and all(not np.array_equal(selected_cage.atoms.positions, m.atoms.positions)
                                         for _, m in candidates):
        candidates.append(('cage_candidate', selected_cage))
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
            centered = minimum.atoms.positions - minimum.atoms.positions.mean(axis=0)
            extent = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(centered))
            distances = np.linalg.norm(centered[:, None] - centered[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            row = dict(status='completed', energy_eV=energy, fmax_eV_A=fmax,
                principal_rms_extents_A=extent.tolist(), minimum_distance_A=float(distances.min()),
                physical_cage_review=('required' if checks[0]['graph_cage_candidate'] else 'not_a_cage'),
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
        selected_cage_index=selected_index, first_stored_joint_index=first_joint_index,
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
