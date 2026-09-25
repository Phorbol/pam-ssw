#!/usr/bin/env python3
"""Run the frozen four-trajectory full-pair LJ pilot; PES work needs --execute."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
PLAN = HERE / 'lj-pilot-plan.md'
REFERENCES = HERE / 'references'
ORACLE = ROOT / 'research/ga_ssw/full_pair_lj.py'
SEEDS = (25092501, 25092502)
REFERENCE_ENERGY = {38: -173.928427, 55: -279.248470}
EPSILON_EV, SIGMA_A = 1.0, 2.7
HIT_ENERGY_TOL = 0.001 * EPSILON_EV
RADIUS_A = 5.5 * SIGMA_A
SEARCH_CAP_TOTAL, WALL_CAP_TOTAL, FRESH_CAP_TOTAL = 2_000_000, 80 * 60, 8


def load_ledger():
    path = ROOT / 'research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py'
    spec = importlib.util.spec_from_file_location('lj_pilot_ledger', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load existing serializer: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def uniform_volume_cluster(n: int, seed: int):
    """IID points uniform in a ball; no overlap rejection or confinement."""
    from ase import Atoms
    children = np.random.SeedSequence(seed).spawn(2)
    init_rng, search_seed = np.random.default_rng(children[0]), children[1]
    direction = init_rng.normal(size=(n, 3))
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm == 0):
        raise FloatingPointError('zero random direction; preserve input failure, do not resample')
    direction /= norm[:, None]
    radius = init_rng.random(n) ** (1.0 / 3.0) * RADIUS_A
    return Atoms(f'Ar{n}', positions=direction * radius[:, None]), search_seed


def geometry_diagnostics(atoms):
    distances = atoms.get_all_distances()
    pair = distances[np.triu_indices(len(atoms), 1)]
    nearest = np.where(np.eye(len(atoms), dtype=bool), np.inf, distances).min(axis=1)
    centered = atoms.positions - atoms.positions.mean(axis=0)
    return {'minimum_pair_distance_A': float(pair.min()),
            'nearest_neighbor_p10_A': float(np.quantile(nearest, 0.1)),
            'nearest_neighbor_median_A': float(np.median(nearest)),
            'nearest_neighbor_max_A': float(nearest.max()),
            'radius_of_gyration_A': float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))),
            'diameter_A': float(pair.max())}


def pair_distance_rms(atoms, reference):
    tri = np.triu_indices(len(atoms), 1)
    a = np.sort(atoms.get_all_distances()[tri])
    b = np.sort(reference.get_all_distances()[tri])
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_settings():
    from ase import units
    from pamssw.standalone import RecoveredRotationSettings, SSWConfig
    config = SSWConfig(width=0.6, rotation_bias=1.0, max_gaussians=14,
        temperature_K=0.8 / units.kB, fmax=0.01, bias_fmax=0.1,
        relax_steps=1000, fd_step=0.001, rotation_hvp=39, rotation_tol=0.02,
        forward_force=0.1, direction_sampling='global',
        rotation_solver='broyden-euclidean', cluster_frame='direction_only',
        quench_optimizer='safe-lbfgs-total', lbfgs_memory=500,
        rotation_exit_policy='force_or_budget')
    rotation = RecoveredRotationSettings(pre_rotmax=5, rotmax=15,
        pre_ftol=0.2, ftol=0.02, metric='euclidean', max_force_calls=40)
    return config, rotation


class BoundedSurface:
    """ASE surface with per-run and campaign request/time bounds."""
    def __init__(self, calculator, *, arm_cap, arm_deadline, total_deadline):
        from pamssw.standalone import ASESurface

        class CountedASESurface(ASESurface):
            pass

        self._surface = CountedASESurface(calculator)
        self.arm_cap, self.arm_deadline, self.total_deadline = arm_cap, arm_deadline, total_deadline
        self.denials, self.boundary = 0, None

    @property
    def requests(self):
        return self._surface.requests

    def evaluate(self, atoms):
        if self.requests >= self.arm_cap:
            self.denials, self.boundary = self.denials + 1, 'arm_request_cap'
            raise RuntimeError(self.boundary)
        now = time.monotonic()
        if now >= self.total_deadline:
            self.denials, self.boundary = self.denials + 1, 'campaign_wall_cap'
            raise RuntimeError(self.boundary)
        if now >= self.arm_deadline:
            self.denials, self.boundary = self.denials + 1, 'arm_wall_cap'
            raise RuntimeError(self.boundary)
        return self._surface.evaluate(atoms)


def first_energy_candidate(minima, n):
    for index, minimum in enumerate(minima):
        if qualifies_energy_candidate(minimum, n):
            return index, minimum
    return None


def qualifies_energy_candidate(minimum, n):
    threshold = REFERENCE_ENERGY[n] + HIT_ENERGY_TOL
    return (minimum.converged and np.isfinite(minimum.energy) and
            np.isfinite(minimum.max_force) and minimum.max_force <= 0.01 and
            minimum.energy <= threshold)


def run_one(n, seed, out, config, rotation, total_deadline, fresh_total, ledger,
            compact_observer=False):
    from ase import Atoms
    from ase.io import write
    from pamssw.standalone import ASESurface, run_ssw
    from pamssw.standalone.paper_reference import InitialQuenchError
    from research.ga_ssw.full_pair_lj import FullPairLJ

    folder = out / f'lj{n}-seed{seed}'
    search_cap, wall_cap, step_cap = ((200_000, 600, 500) if n == 55
                                      else (800_000, 1800, 2000))
    started = time.monotonic()
    arm_deadline = started + wall_cap
    atoms, search_seed = uniform_volume_cluster(n, seed)
    input_path = folder / 'initial.extxyz'
    write(input_path, atoms)
    target = Atoms(f'Ar{n}', positions=np.loadtxt(REFERENCES / f'lj{n}.points') * SIGMA_A)
    surface = BoundedSurface(FullPairLJ(epsilon=EPSILON_EV, sigma=SIGMA_A),
        arm_cap=search_cap, arm_deadline=arm_deadline, total_deadline=total_deadline)
    rng = np.random.default_rng(search_seed)
    checkpoint = None
    latest = None
    result = None
    hit = None
    callback_seconds = 0.0
    saved_minima = 0
    step_log = folder / 'outer-steps.jsonl'
    minima_dir = folder / 'minima'
    minima_dir.mkdir()

    def save_new_minimum(index, minimum):
        nonlocal saved_minima
        if index != saved_minima:
            raise RuntimeError(f'nonsequential minimum index {index}, expected {saved_minima}')
        write(minima_dir / f'minimum-{index:04d}.extxyz', minimum.atoms)
        row = {'index': index, 'energy_eV': float(minimum.energy),
            'fmax_eV_A': float(minimum.max_force), 'converged': bool(minimum.converged),
            **geometry_diagnostics(minimum.atoms)}
        saved_minima += 1
        return row

    def save_new_minima(snapshot):
        new = []
        for index in range(saved_minima, len(snapshot.minima)):
            minimum = snapshot.minima[index]
            new.append(save_new_minimum(index, minimum))
        return new

    row = {'n': n, 'seed': seed, 'status': 'started', 'search_cap': search_cap,
        'wall_cap_seconds': wall_cap, 'outer_step_cap': step_cap,
        'observer_mode': 'compact' if compact_observer else 'checkpoint',
        'initial_input': str(input_path), 'radius_A': RADIUS_A,
        'initialization': 'independent iid uniform-volume points in a sphere; no rejection',
        'rng_streams': 'SeedSequence(seed).spawn(): child 0 positions, child 1 SSW',
        'reference_energy_eV': REFERENCE_ENERGY[n], 'energy_candidate_tolerance_eV': HIT_ENERGY_TOL,
        'settings': {'ssw_config': asdict(config), 'recovered_rotation': asdict(rotation),
                     'mc': None, 'ls': None, 'potential': 'full-pair nonperiodic LJ'}}

    try:
        # Separate starting quench lets an already-qualified start stop before
        # any outer attempt. The no-op callback requests a checkpoint only.
        initial_result = run_ssw(atoms.copy(), surface, steps=0, config=config,
            rng=rng, recovered_rotation=rotation,
            checkpoint_callback=lambda _state: False)
        checkpoint = initial_result.checkpoint
        if checkpoint is None:
            raise RuntimeError('starting quench returned no zero-step checkpoint')
        latest = checkpoint
        initial = checkpoint.initial
        new_initial = save_new_minima(checkpoint)
        hit_match = first_energy_candidate(checkpoint.minima, n)
        if hit_match is not None:
            index, minimum = hit_match
            hit = {'minimum_index': index, 'step': -1, 'source': 'initial_quench',
                   'cumulative_search_requests': int(checkpoint.evaluation_requests),
                   'energy_eV': float(minimum.energy), 'fmax_eV_A': float(minimum.max_force)}
        with step_log.open('a') as stream:
            stream.write(json.dumps({'step': -1, 'status': 'initial_quench',
                'requests': int(initial.evaluation_requests),
                'cumulative_requests': int(checkpoint.evaluation_requests),
                'energy_eV': float(initial.energy), 'fmax_eV_A': float(initial.max_force),
                'converged': bool(initial.converged), 'new_minima': new_initial,
                'first_hit': hit}, allow_nan=False) + '\n')
            stream.flush()

        if hit is None:
            def record_outer_event(record, new_minima, cumulative_requests,
                                   current_energy, best_energy):
                nonlocal hit
                if new_minima:
                    minimum_index, minimum, _row = new_minima[0]
                    if qualifies_energy_candidate(minimum, n):
                        hit = {'minimum_index': minimum_index,
                            'step': int(record.index),
                            'cumulative_search_requests': int(cumulative_requests),
                            'energy_eV': float(minimum.energy),
                            'fmax_eV_A': float(minimum.max_force)}
                with step_log.open('a') as stream:
                    stream.write(json.dumps({'step': int(record.index),
                        'status': str(record.status), 'accepted': bool(record.accepted),
                        'step_requests': int(record.evaluation_requests),
                        'cumulative_requests': int(cumulative_requests),
                        'landing_energy_eV': (None if record.landing is None else
                                              float(record.landing.energy)),
                        'current_energy_eV': float(current_energy),
                        'best_energy_eV': float(best_energy),
                        'new_minima': [row for _, _, row in new_minima],
                        'first_hit': hit},
                        allow_nan=False) + '\n')
                    stream.flush()
                return hit is not None

            def after_outer_step(snapshot):
                nonlocal latest, callback_seconds
                tick = time.monotonic()
                try:
                    latest = snapshot
                    new_rows = save_new_minima(snapshot)
                    new_start = len(snapshot.minima) - len(new_rows)
                    new_minima = [(new_start + offset, snapshot.minima[new_start + offset], row)
                                  for offset, row in enumerate(new_rows)]
                    return record_outer_event(snapshot.records[-1], new_minima,
                        snapshot.evaluation_requests, snapshot.current_energy, snapshot.best.energy)
                finally:
                    callback_seconds += time.monotonic() - tick

            def after_compact_progress(progress):
                nonlocal callback_seconds
                tick = time.monotonic()
                try:
                    if progress.kind == 'initial':
                        # run_one has already ingested the zero-step initial minimum;
                        # resumed call-start carries new_minimum=None by contract.
                        if progress.new_minimum is not None:
                            raise RuntimeError('compact resume unexpectedly emitted a fresh initial minimum')
                        return False
                    if progress.kind != 'outer_step' or progress.step is None:
                        raise RuntimeError(f'unexpected compact progress event: {progress.kind!r}')
                    new_minima = []
                    if progress.new_minimum is not None:
                        minimum_index = saved_minima
                        minimum_row = save_new_minimum(minimum_index, progress.new_minimum)
                        new_minima.append((minimum_index, progress.new_minimum, minimum_row))
                    return record_outer_event(progress.step, new_minima,
                        progress.evaluation_requests, progress.current_energy, progress.best.energy)
                finally:
                    callback_seconds += time.monotonic() - tick

            # One driver call avoids repeated checkpoint restoration. Each
            # completed outer step still reports through the safe callback.
            if compact_observer:
                result = run_ssw(atoms.copy(), surface, steps=step_cap, config=config,
                    rng=rng, checkpoint=checkpoint, progress_callback=after_compact_progress,
                    recovered_rotation=rotation)
            else:
                result = run_ssw(atoms.copy(), surface, steps=step_cap, config=config,
                    rng=rng, checkpoint=checkpoint, checkpoint_callback=after_outer_step,
                    recovered_rotation=rotation)
            latest = result.checkpoint or latest
        row['status'] = ('first_hit' if hit is not None else
                         (result.status if result is not None else 'initial_quench_failed'))
    except InitialQuenchError as error:
        failed = error.result
        row.update(status='initial_quench_failed', error=repr(error),
            initial_quench_requests=int(failed.evaluation_requests),
            initial_energy_eV=float(failed.energy), initial_fmax_eV_A=float(failed.max_force),
            initial_converged=bool(failed.converged))
        with step_log.open('a') as stream:
            stream.write(json.dumps({'step': -1, 'status': row['status'],
                'requests': int(failed.evaluation_requests),
                'cumulative_requests': int(surface.requests), 'error': repr(error)}) + '\n')
            stream.flush()
    except Exception as error:
        row.update(status='exception', error=repr(error), traceback=traceback.format_exc())
        with step_log.open('a') as stream:
            stream.write(json.dumps({'step': None, 'status': 'exception',
                'search_requests': int(surface.requests),
                'checkpoint_requests': (None if latest is None else int(latest.evaluation_requests)),
                'uncheckpointed_requests': (None if latest is None else
                    int(surface.requests - latest.evaluation_requests)),
                'boundary': surface.boundary, 'error': repr(error)}) + '\n')
            stream.flush()

    elapsed = time.monotonic() - started
    if latest is not None:
        best = latest.best
        write(folder / 'best.extxyz', best.atoms)
        row.update(completed_outer_attempts=len(latest.records),
            checkpoint_requests=int(latest.evaluation_requests),
            best_energy_eV=float(best.energy), best_fmax_eV_A=float(best.max_force),
            final_best_geometry=str(folder / 'best.extxyz'),
            final_best_geometry_diagnostics=geometry_diagnostics(best.atoms))
        if hit is not None:
            row['candidate_geometry'] = str(folder / 'candidate.extxyz')
            write(folder / 'candidate.extxyz', latest.minima[hit['minimum_index']].atoms)
            hit['pair_distance_rms_to_reference_A'] = pair_distance_rms(
                latest.minima[hit['minimum_index']].atoms, target)
            row['first_hit'] = hit
    row.update(search_requests=int(surface.requests), denials=int(surface.denials),
        boundary=surface.boundary, wall_seconds=float(elapsed),
        callback_function_seconds=float(callback_seconds),
        requests_equal_checkpoint=(latest is not None and
            int(latest.evaluation_requests) == int(surface.requests)))

    # Fresh checks have their own calculator and stay outside search cost.
    from pamssw.standalone import ASESurface
    fresh = ASESurface(FullPairLJ(epsilon=EPSILON_EV, sigma=SIGMA_A))
    candidates = []
    if latest is not None:
        if hit is not None:
            candidates.append(('candidate', latest.minima[hit['minimum_index']]))
        candidates.append(('final_best', latest.best))
    checks = []
    for label, minimum in candidates:
        if fresh_total[0] >= FRESH_CAP_TOTAL or time.monotonic() >= total_deadline:
            break
        try:
            energy, forces = fresh.evaluate(minimum.atoms)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            checks.append({'label': label, 'energy_eV': float(energy),
                'energy_error_eV': float(energy - minimum.energy), 'fmax_eV_A': fmax,
                'force_qualified': bool(fmax <= config.fmax),
                'energy_candidate': bool(energy <= REFERENCE_ENERGY[n] + HIT_ENERGY_TOL),
                'pair_distance_rms_to_reference_A': pair_distance_rms(minimum.atoms, target)})
            fresh_total[0] += 1
        except Exception as error:
            checks.append({'label': label, 'error': repr(error)})
    row['fresh_checks'], row['fresh_requests'] = checks, int(fresh.requests)
    row['wall_budget_remaining_campaign_seconds'] = max(0.0, total_deadline - time.monotonic())
    ledger.dump(folder / 'summary.json', row)
    return row


def preflight(output: Path):
    """Validate paths, settings, references and seeded inputs without PES calls."""
    if output.exists():
        raise FileExistsError(output)
    for path in (PLAN, ORACLE, REFERENCES / 'lj38.points', REFERENCES / 'lj55.points'):
        if not path.is_file():
            raise FileNotFoundError(path)
    qualifications = json.loads((HERE / 'lj-reference-qualification.json').read_text())
    for n in (38, 55):
        row = next(item for item in qualifications if item['n'] == n)
        points = np.loadtxt(REFERENCES / f'lj{n}.points')
        if not row['qualified'] or points.shape != (n, 3) or not np.isfinite(points).all():
            raise ValueError(f'LJ{n} reference is not ready')
    config, rotation = make_settings()
    # Construction checks API/settings only; no calculator is attached/evaluated.
    from research.ga_ssw.full_pair_lj import FullPairLJ
    FullPairLJ(epsilon=EPSILON_EV, sigma=SIGMA_A)
    inputs = []
    for n in (55, 38):
        for seed in SEEDS:
            atoms, _ = uniform_volume_cluster(n, seed)
            inputs.append({'n': n, 'seed': seed, 'atoms': len(atoms),
                           'pbc': atoms.pbc.tolist(),
                           'max_radius_A': float(np.linalg.norm(atoms.positions, axis=1).max())})
    print(json.dumps({'status': 'preflight_passed_zero_PES', 'root': str(ROOT),
        'plan': str(PLAN), 'output_available': str(output),
        'reference_checks': qualifications, 'ssw_config': asdict(config),
        'recovered_rotation': asdict(rotation), 'seeded_inputs': inputs,
        'PES_requests': 0}, indent=2))


def execute(output: Path, compact_observer=False):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    ledger = load_ledger()
    shutil.copy2(Path(__file__).resolve(), output / 'run_lj_pilot.py')
    shutil.copy2(PLAN, output / 'lj-pilot-plan.md')
    shutil.copy2(ORACLE, output / 'full_pair_lj.py')
    refs_out = output / 'references'
    refs_out.mkdir()
    for n in (38, 55):
        shutil.copy2(REFERENCES / f'lj{n}.points', refs_out / f'lj{n}.points')

    config, rotation = make_settings()
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    ledger.dump(output / 'execution.json', {'git_head': git('rev-parse', 'HEAD'),
        'core_tree': git('rev-parse', 'HEAD:pamssw'),
        'observer_mode': 'compact' if compact_observer else 'checkpoint',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        'search_cap_total': SEARCH_CAP_TOTAL, 'wall_cap_total_seconds': WALL_CAP_TOTAL,
        'fresh_cap_total': FRESH_CAP_TOTAL})
    ledger.dump(output / 'effective-plan.json', {'settings': {'ssw_config': config,
        'recovered_rotation': rotation, 'mc': None, 'ls': None},
        'observer_mode': 'compact' if compact_observer else 'checkpoint',
        'trajectory_order': [{'n': n, 'seed': seed} for n in (55, 38) for seed in SEEDS],
        'reference_energy_eV': REFERENCE_ENERGY, 'epsilon_eV': EPSILON_EV,
        'sigma_A': SIGMA_A, 'radius_A': RADIUS_A, 'search_cap_total': SEARCH_CAP_TOTAL,
        'wall_cap_total_seconds': WALL_CAP_TOTAL, 'fresh_cap_total': FRESH_CAP_TOTAL})

    started = time.monotonic()
    deadline = started + WALL_CAP_TOTAL
    search_total, fresh_total, rows = 0, [0], []
    for n in (55, 38):
        for seed in SEEDS:
            folder = output / f'lj{n}-seed{seed}'
            folder.mkdir()
            if n == 38 and not any(r.get('n') == 55 and r.get('status') == 'first_hit' for r in rows):
                row = {'n': n, 'seed': seed, 'status': 'not_run_positive_control_gate',
                       'search_requests': 0, 'fresh_requests': 0}
                ledger.dump(folder / 'summary.json', row)
            elif search_total >= SEARCH_CAP_TOTAL or time.monotonic() >= deadline:
                row = {'n': n, 'seed': seed, 'status': 'not_run_campaign_cap',
                       'search_requests': 0, 'fresh_requests': 0}
            else:
                try:
                    row = run_one(n, seed, output, config, rotation, deadline, fresh_total,
                                  ledger, compact_observer=compact_observer)
                except Exception as error:
                    row = {'n': n, 'seed': seed, 'status': 'exception',
                           'search_requests': 0, 'fresh_requests': 0,
                           'error': repr(error), 'traceback': traceback.format_exc()}
                    ledger.dump(folder / 'summary.json', row)
                search_total += int(row['search_requests'])
            rows.append(row)
            ledger.dump(output / 'summary.json', {'trajectories': rows,
                'search_requests_total': search_total, 'fresh_requests_total': fresh_total[0],
                'wall_seconds_total': time.monotonic() - started,
                'status': 'running' if len(rows) < 4 else 'complete_or_censored'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observer', choices=('checkpoint', 'compact'), default='checkpoint',
        help='outer-step reporting mode; checkpoint preserves legacy default')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.preflight:
        preflight(output)
    elif args.execute:
        execute(output, compact_observer=(args.observer == 'compact'))
    else:
        print('Runner prepared. Use --preflight for zero-PES checks; --execute starts PES work.')


if __name__ == '__main__':
    main()
