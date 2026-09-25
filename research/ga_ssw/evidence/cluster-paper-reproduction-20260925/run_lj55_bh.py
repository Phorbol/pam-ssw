#!/usr/bin/env python3
"""Bounded ASE 3.26 LJ55 BasinHopping arm; PES calls require --execute."""
from __future__ import annotations

import argparse
import importlib.util
import json
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
INPUTS = HERE / 'runs'
SEEDS = (25092501, 25092502)
REFERENCE = -279.248470
EPSILON, SIGMA = 1.0, 2.7
FMAX, KT, DR = 0.01, 0.8, 0.38 * SIGMA
RELAX_STEPS, HISTORY = 1000, 500
ARM_REQUESTS, ARM_WALL, OUTER_STEPS = 200_000, 600, 5000
CAMPAIGN_REQUESTS, CAMPAIGN_WALL, FRESH_LIMIT = 400_000, 1200, 4


def serializer():
    path = ROOT / 'research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py'
    spec = importlib.util.spec_from_file_location('lj55_bh_ledger', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def append(ledger, path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(ledger._jsonable(row), allow_nan=False) + '\n')


def run_track(seed, out, ledger, campaign_deadline, campaign_cost, fresh_cost):
    from ase import __version__ as ase_version
    from ase.io import read, write
    from ase.optimize.basin import BasinHopping
    from pamssw.standalone import ASESurface
    from pamssw.standalone.surface import quench, SurfaceCalculator
    from research.ga_ssw.full_pair_lj import FullPairLJ
    if ase_version != '3.26.0':
        raise RuntimeError(f'expected isolated ASE 3.26.0, imported {ase_version}')

    folder = out / f'lj55-seed{seed}'
    folder.mkdir()
    initial_path = INPUTS / f'lj55-seed{seed:08d}' / 'initial.extxyz'
    if not initial_path.exists():
        initial_path = INPUTS / f'lj55-seed{seed}' / 'initial.extxyz'
    atoms = read(initial_path)
    if len(atoms) != 55 or atoms.pbc.any():
        raise ValueError(f'invalid LJ55 input: {initial_path}')
    started = time.monotonic()
    arm_deadline = min(started + ARM_WALL, campaign_deadline)

    class BoundedSurface(ASESurface):
        boundary = None

        def evaluate(self, candidate):
            if self.requests >= ARM_REQUESTS or campaign_cost[0] >= CAMPAIGN_REQUESTS:
                self.boundary = 'request_cap'
                raise RuntimeError(self.boundary)
            now = time.monotonic()
            if now >= arm_deadline or now >= campaign_deadline:
                self.boundary = 'wall_cap'
                raise RuntimeError(self.boundary)
            campaign_cost[0] += 1
            return super().evaluate(candidate)

    surface = BoundedSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA))
    minima_path = folder / 'quench-records.jsonl'
    best = None
    candidate = None
    quench_count = 0
    status, error_text = 'started', None

    class CandidateFound(Exception):
        pass

    class SafeTotalOptimizer:
        def __init__(self, optimizable, *, logfile=None):
            self.optimizable = optimizable

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def run(self, fmax=None, steps=None):
            nonlocal best, candidate, quench_count
            before = surface.requests
            try:
                result = quench(self.optimizable.atoms, surface, fmax=FMAX,
                                steps=RELAX_STEPS, optimizer='safe-lbfgs-total',
                                lbfgs_memory=HISTORY)
            except Exception as error:
                quench_count += 1
                append(ledger, minima_path, {'quench': quench_count, 'energy_eV': None,
                    'fmax_eV_A': None, 'search_requests': surface.requests - before,
                    'cumulative_search_requests': surface.requests, 'converged': False,
                    'new_best': False, 'error': repr(error)})
                raise
            requests = surface.requests - before
            if result.evaluation_requests != requests:
                raise AssertionError('quench and shared surface request counts differ')
            quench_count += 1
            new_best = bool(result.converged and (best is None or result.energy < best.energy))
            if new_best:
                best = result
                write(folder / 'best.extxyz', best.atoms)
            record = {'quench': quench_count, 'energy_eV': result.energy,
                      'fmax_eV_A': result.max_force, 'search_requests': requests,
                      'cumulative_search_requests': surface.requests,
                      'converged': result.converged, 'optimizer_steps': result.optimizer_steps,
                      'new_best': new_best}
            append(ledger, minima_path, record)
            if (result.converged and result.max_force <= FMAX and
                    result.energy <= REFERENCE + 0.001 * EPSILON):
                candidate = result
                write(folder / 'candidate.extxyz', candidate.atoms)
                raise CandidateFound('energy_and_force_candidate')
            if not result.converged:
                raise RuntimeError('Safe-total quench did not converge')
            self.optimizable.atoms.set_positions(result.atoms.positions)
            return True

    def optimizer_factory(optimizable, logfile=None):
        return SafeTotalOptimizer(optimizable, logfile=logfile)

    np.random.seed(seed)  # ASE 3.26 uses NumPy's global RandomState.
    try:
        atoms.calc = SurfaceCalculator(surface)
        basin = BasinHopping(atoms, temperature=KT * EPSILON, dr=DR, fmax=FMAX,
                             logfile=None, trajectory=None, optimizer=optimizer_factory,
                             optimizer_logfile=None, local_minima_trajectory=None)
        basin.run(OUTER_STEPS)
        status = 'completed'
    except CandidateFound:
        status = 'first_energy_candidate'
    except Exception as error:
        status, error_text = 'exception', repr(error)

    fresh_rows = []
    for name, result in (('candidate', candidate), ('best', best)):
        if result is None or fresh_cost[0] >= FRESH_LIMIT:
            continue
        fresh_surface = ASESurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA))
        try:
            energy, forces = fresh_surface.evaluate(result.atoms)
            fresh_cost[0] += fresh_surface.requests
            fresh_rows.append({'kind': name, 'energy_eV': energy,
                'fmax_eV_A': float(np.linalg.norm(forces, axis=1).max()),
                'requests': fresh_surface.requests})
        except Exception as error:
            fresh_cost[0] += fresh_surface.requests
            fresh_rows.append({'kind': name, 'error': repr(error),
                               'requests': fresh_surface.requests})
    row = {'n': 55, 'seed': seed, 'status': status, 'error': error_text,
           'input': str(initial_path), 'ase_version': ase_version,
           'temperature_eV': KT * EPSILON, 'dr_A': DR, 'fmax_eV_A': FMAX,
           'relax_steps': RELAX_STEPS, 'lbfgs_memory': HISTORY,
           'outer_step_cap': OUTER_STEPS, 'search_cap': ARM_REQUESTS,
           'search_requests': surface.requests, 'campaign_search_requests': campaign_cost[0],
           'quench_records': quench_count, 'boundary': surface.boundary,
           'fresh_checks': fresh_rows, 'fresh_requests_total': fresh_cost[0],
           'wall_seconds': time.monotonic() - started,
           'candidate_geometry': str(folder / 'candidate.extxyz') if candidate else None,
           'best_geometry': str(folder / 'best.extxyz') if best else None}
    ledger.dump(folder / 'summary.json', row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='run bounded LJ55 PES trajectories')
    parser.add_argument('--output', type=Path, default=HERE / 'bh-runs')
    args = parser.parse_args()
    if not args.execute:
        parser.error('PES execution is opt-in; pass --execute')
    out = args.output.resolve()
    out.mkdir(parents=False, exist_ok=False)
    ledger = serializer()
    shutil.copy2(__file__, out / 'run_lj55_bh.py')
    shutil.copy2(HERE/'lj55-bh-plan.md',out/'plan.md')
    ledger.dump(out/'execution.json',{'git_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'python':sys.executable})
    started = time.monotonic()
    campaign_deadline = started + CAMPAIGN_WALL
    campaign_cost, fresh_cost = [0], [0]
    rows = []
    for seed in SEEDS:
        if campaign_cost[0] >= CAMPAIGN_REQUESTS or time.monotonic() >= campaign_deadline:
            row = {'n': 55, 'seed': seed, 'status': 'not_run_campaign_cap',
                   'search_requests': 0, 'fresh_requests_total': fresh_cost[0]}
            (out / f'lj55-seed{seed}').mkdir()
            ledger.dump(out / f'lj55-seed{seed}' / 'summary.json', row)
        else:
            try:
                row = run_track(seed, out, ledger, campaign_deadline, campaign_cost, fresh_cost)
            except Exception as error:
                track_dir = out / f'lj55-seed{seed}'
                track_dir.mkdir(exist_ok=True)
                row = {'n': 55, 'seed': seed, 'status': 'setup_exception',
                       'error': repr(error), 'traceback': traceback.format_exc(),
                       'fresh_requests_total': fresh_cost[0],
                       'campaign_search_requests_at_failure': campaign_cost[0]}
                ledger.dump(track_dir / 'summary.json', row)
        rows.append(row)
        ledger.dump(out / 'summary.json', {'trajectories': rows,
            'campaign_search_requests': campaign_cost[0], 'fresh_requests_total': fresh_cost[0],
            'wall_seconds': time.monotonic() - started,
            'status': 'complete_or_censored' if len(rows) == len(SEEDS) else 'running'})


if __name__ == '__main__':
    main()
