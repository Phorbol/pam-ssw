#!/usr/bin/env python3
"""One-step matched LJ Gaussian-policy probe; PES work requires --execute."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pamssw.standalone import ASESurface

EVIDENCE = ROOT / 'research/ga_ssw/evidence/cluster-paper-reproduction-20260925'
GENERATOR = EVIDENCE / 'runs/run_lj_pilot.py'
REFERENCE_QUALIFICATION = EVIDENCE / 'lj-reference-qualification.json'
POLICIES = ('forward_default', 'pam_curvature_height_width')
SIZES = (38, 55)
SEEDS = (25092501, 25092502)
ARM_REQUEST_CAP = 2500
CAMPAIGN_WALL_SECONDS = 240
FRESH_REQUEST_CAP_TOTAL = 16
EPSILON_EV, SIGMA_A = 1.0, 2.7


def _load_generator():
    if not GENERATOR.is_file():
        raise FileNotFoundError(GENERATOR)
    spec = importlib.util.spec_from_file_location('lj_gaussian_probe_generator', GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load fixed LJ input/settings source: {GENERATOR}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _settings(generator):
    config, rotation = generator.make_settings()
    config = replace(config, direction_sampling='paper')
    return config, rotation


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _dump(path: Path, value):
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + '\n')


def _serial(value):
    from research.ga_ssw.compare_vc_arms import serial
    return _json_safe(serial(value))


def _check_connectivity(atoms):
    positions = np.asarray(atoms.positions, dtype=float)
    n = len(positions)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    tri = np.triu_indices(n, 1)
    minimum = float(distances[tri].min()) if n > 1 else None
    reports = {}
    for multiplier in (1.3, 1.5):
        cutoff = multiplier * SIGMA_A
        unseen = set(range(n))
        components = 0
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                i = stack.pop()
                neighbors = [j for j in tuple(unseen) if distances[i, j] <= cutoff]
                for j in neighbors:
                    unseen.remove(j)
                    stack.append(j)
        reports[f'{multiplier:.1f}_sigma'] = {
            'cutoff_A': cutoff,
            'components': components,
            'connected': components == 1,
        }
    return {'minimum_pair_distance_A': minimum, 'thresholds': reports}


class CountedASESurface(ASESurface):
    """ASE surface that denies calls past arm, campaign-time, or fresh caps."""
    def __init__(self, calculator, *, cap, deadline, fresh_total=None, case_cost=None):
        super().__init__(calculator)
        self.cap = int(cap)
        self.deadline = float(deadline)
        self.fresh_total = fresh_total
        self.case_cost = case_cost
        self.denials = 0
        self.boundary = None

    def evaluate(self, atoms):
        boundary = None
        if self.requests >= self.cap:
            boundary = 'request_cap'
        elif self.fresh_total is not None and self.fresh_total[0] >= FRESH_REQUEST_CAP_TOTAL:
            boundary = 'campaign_fresh_cap'
        elif time.monotonic() >= self.deadline:
            boundary = 'campaign_wall_cap'
        if boundary is not None:
            self.denials += 1
            self.boundary = boundary
            if self.case_cost is not None:
                prefix = 'fresh' if self.fresh_total is not None else 'search'
                self.case_cost[f'{prefix}_denials'] = self.case_cost.get(f'{prefix}_denials', 0) + 1
                self.case_cost[f'{prefix}_boundary'] = boundary
            raise RuntimeError(boundary)
        before = self.requests
        try:
            return super().evaluate(atoms)
        finally:
            delta = self.requests - before
            if self.case_cost is not None:
                key = 'fresh_requests' if self.fresh_total is not None else 'search_requests'
                self.case_cost[key] = self.case_cost.get(key, 0) + delta
            if self.fresh_total is not None:
                self.fresh_total[0] += delta


def _fresh_check(atoms, *, label, deadline, fresh_total, case_cost):
    from pamssw.standalone import ASESurface
    from research.ga_ssw.full_pair_lj import FullPairLJ

    surface = CountedASESurface(FullPairLJ(epsilon=EPSILON_EV, sigma=SIGMA_A),
        cap=1, deadline=deadline, fresh_total=fresh_total, case_cost=case_cost)
    row = {'label': label}
    try:
        energy, forces = surface.evaluate(atoms)
        row.update(status='evaluated', energy_eV=float(energy),
            fmax_eV_A=float(np.linalg.norm(forces, axis=1).max()),
            connectivity=_check_connectivity(atoms))
    except Exception as error:
        row.update(status='failed', error=repr(error))
    row.update(requests=surface.requests, denials=surface.denials,
               boundary=surface.boundary)
    return row


def _mode_record(record):
    if record is None or not getattr(record, 'climb', None):
        return None
    return record.climb[0]


def _first_mode_atoms(input_atoms, mode_record):
    if not isinstance(mode_record, dict):
        return None
    center, direction = mode_record.get('center'), mode_record.get('direction')
    width = mode_record.get('width')
    if center is None or direction is None or width is None:
        return None
    from ase import Atoms
    atoms = input_atoms.copy()
    center = np.asarray(center, dtype=float).reshape((-1, 3))
    direction = np.asarray(direction, dtype=float).reshape((-1, 3))
    if center.shape != atoms.positions.shape or direction.shape != atoms.positions.shape:
        return None
    atoms.positions[:] = center + float(width) * direction
    return atoms


def _run_case(generator, *, n, seed, policy_name, output_dir, deadline,
              fresh_total, case_cost, config, rotation):
    from ase.io import write
    from pamssw.standalone import run_ssw
    from pamssw.standalone.paper_reference import InitialQuenchError
    from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
    from research.ga_ssw.full_pair_lj import FullPairLJ

    folder = output_dir / f'lj{n}-seed{seed}' / policy_name
    folder.mkdir(parents=True)
    atoms, search_seed = generator.uniform_volume_cluster(n, seed)
    write(folder / 'input.extxyz', atoms)
    policy = None if policy_name == 'forward_default' else PAMCurvatureGaussian()
    effective = {
        'n': n, 'seed': seed, 'policy': policy_name,
        'ssw_config': asdict(config), 'recovered_rotation': asdict(rotation),
        'gaussian_policy': None if policy is None else policy.parameters(),
        'steps': 1, 'arm_request_cap_including_initial': ARM_REQUEST_CAP,
        'search_rng_seed_sequence': {
            'entropy': search_seed.entropy, 'spawn_key': list(search_seed.spawn_key),
            'pool_size': search_seed.pool_size,
        },
        'potential': {'kind': 'FullPairLJ', 'epsilon_eV': EPSILON_EV,
                      'sigma_A': SIGMA_A, 'cutoff': None, 'pbc': False},
    }
    _dump(folder / 'effective-config.json', effective)
    surface = CountedASESurface(FullPairLJ(epsilon=EPSILON_EV, sigma=SIGMA_A),
        cap=ARM_REQUEST_CAP, deadline=deadline, case_cost=case_cost)
    started = time.monotonic()
    result = None
    initial_result = None
    error_text = None
    status = 'started'
    try:
        result = run_ssw(atoms.copy(), surface, steps=1, config=config,
            rng=np.random.default_rng(search_seed), recovered_rotation=rotation,
            gaussian_policy=policy)
        initial_result = result.initial
        status = result.status
    except InitialQuenchError as error:
        initial_result = error.result
        status = 'initial_quench_failed'
        error_text = repr(error)
    except Exception as error:
        status = 'exception'
        error_text = repr(error)
        (folder / 'traceback.txt').write_text(traceback.format_exc())

    records = () if result is None else result.records
    row_record = records[0] if records else None
    first_mode = _mode_record(row_record)
    if first_mode is not None:
        first_mode = _serial(first_mode)
        _dump(folder / 'first-mode.json', first_mode)
        displaced = _first_mode_atoms(atoms, first_mode)
        if displaced is not None:
            write(folder / 'first-mode.extxyz', displaced)
    if result is not None:
        _dump(folder / 'records.json', _serial(result.records))
    else:
        _dump(folder / 'records.json', [])

    initial_atoms = None if initial_result is None else getattr(initial_result, 'atoms', None)
    landing = None if row_record is None else getattr(row_record, 'landing', None)
    landing_atoms = None if landing is None else getattr(landing, 'atoms', None)
    if initial_atoms is not None:
        write(folder / 'initial.extxyz', initial_atoms)
    if landing_atoms is not None:
        write(folder / 'landing.extxyz', landing_atoms)

    fresh_checks = []
    if initial_atoms is not None:
        fresh_checks.append(_fresh_check(initial_atoms, label='initial',
            deadline=deadline, fresh_total=fresh_total, case_cost=case_cost))
    if landing_atoms is not None:
        fresh_checks.append(_fresh_check(landing_atoms, label='landing',
            deadline=deadline, fresh_total=fresh_total, case_cost=case_cost))

    climbs = () if row_record is None else row_record.climb
    clipping = {
        'width_clamped': sum(bool((event.get('gaussian_policy') or {}).get('width_clamped'))
                             for event in climbs if isinstance(event, dict)),
        'weight_clamped': sum(bool((event.get('gaussian_policy') or {}).get('weight_clamped'))
                              for event in climbs if isinstance(event, dict)),
    }
    elapsed = time.monotonic() - started
    summary = {
        **effective, 'status': status, 'error': error_text,
        'search_requests': surface.requests, 'request_cap': surface.cap,
        'denials': surface.denials, 'boundary': surface.boundary,
        'wall_seconds': elapsed,
        'initial': None if initial_result is None else {
            'energy_eV': getattr(initial_result, 'energy', None),
            'fmax_eV_A': getattr(initial_result, 'max_force', None),
            'converged': getattr(initial_result, 'converged', None),
            'geometry': 'initial.extxyz' if initial_atoms is not None else None,
            'connectivity': None if initial_atoms is None else _check_connectivity(initial_atoms),
        },
        'first_gaussian': first_mode,
        'first_mode_geometry': 'first-mode.extxyz' if (folder / 'first-mode.extxyz').exists() else None,
        'landing': None if landing is None else {
            'energy_eV': getattr(landing, 'energy', None),
            'max_force_eV_A': getattr(landing, 'max_force', None),
            'converged': getattr(landing, 'converged', None),
            'geometry': 'landing.extxyz' if landing_atoms is not None else None,
            'connectivity': None if landing_atoms is None else _check_connectivity(landing_atoms),
        },
        'fresh_checks': fresh_checks,
        'fresh_requests': sum(check['requests'] for check in fresh_checks),
        'clipping_counts': clipping,
        'records_file': 'records.json',
    }
    _dump(folder / 'summary.json', summary)
    return summary


def prepare(output_dir: Path):
    generator = _load_generator()
    config, rotation = _settings(generator)
    payload = _prepare_payload(output_dir, generator, config, rotation)
    payload['status'] = 'prepare_only_passed_zero_PES'
    print(json.dumps(payload, indent=2, allow_nan=False))


def execute(output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    generator = _load_generator()
    config, rotation = _settings(generator)
    prepare_info = _prepare_payload(output_dir, generator, config, rotation)
    output_dir.mkdir(parents=True)
    import shutil
    shutil.copy2(Path(__file__).resolve(), output_dir / Path(__file__).name)
    shutil.copy2(GENERATOR, output_dir / 'input-generator.py')
    import ase
    git_head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    tracked_diff = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet'], check=False).returncode != 0
    _dump(output_dir / 'execution.json', {
        'git_head': git_head, 'tracked_worktree_diff_present': tracked_diff,
        'python': sys.executable, 'ase_version': ase.__version__, 'numpy_version': np.__version__,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol': 'research/ga_ssw/evidence/lj-gaussian-policy-probe-20260927/protocol.md',
        'campaign_wall_seconds': CAMPAIGN_WALL_SECONDS,
        'search_request_cap_per_arm': ARM_REQUEST_CAP,
        'fresh_request_cap_total': FRESH_REQUEST_CAP_TOTAL,
        'source': str(GENERATOR), 'source_sha256': hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        'prepare': prepare_info,
    })
    started = time.monotonic()
    deadline = started + CAMPAIGN_WALL_SECONDS
    fresh_total = [0]
    rows = []
    pairs = [(n, seed) for n in SIZES for seed in SEEDS]
    for pair_index, (n, seed) in enumerate(pairs):
        order = POLICIES if pair_index % 2 == 0 else tuple(reversed(POLICIES))
        for policy_name in order:
            case_dir = output_dir / f'lj{n}-seed{seed}' / policy_name
            case_cost = {'search_requests': 0, 'fresh_requests': 0,
                         'search_denials': 0, 'fresh_denials': 0}
            if time.monotonic() >= deadline:
                case_dir.mkdir(parents=True, exist_ok=True)
                summary = {'n': n, 'seed': seed, 'policy': policy_name,
                    'status': 'not_run_campaign_wall_cap', 'search_requests': 0,
                    'fresh_requests': 0, 'wall_seconds': 0.0,
                    'denials': 0, 'boundary': 'campaign_wall_cap'}
                _dump(case_dir / 'summary.json', summary)
            else:
                try:
                    summary = _run_case(generator, n=n, seed=seed,
                        policy_name=policy_name, output_dir=output_dir,
                        deadline=deadline, fresh_total=fresh_total, case_cost=case_cost,
                        config=config, rotation=rotation)
                except Exception as error:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    summary = {'n': n, 'seed': seed, 'policy': policy_name,
                        'status': 'runner_exception', 'error': repr(error),
                        'traceback': traceback.format_exc(),
                        **case_cost, 'wall_seconds': 0.0}
                    _dump(case_dir / 'summary.json', summary)
            rows.append(summary)
            _dump(output_dir / 'summary.json', {
                'status': 'running', 'campaign_wall_seconds': time.monotonic() - started,
                'search_requests_total': sum(row.get('search_requests', 0) for row in rows),
                'fresh_requests_total': sum(row.get('fresh_requests', 0) for row in rows),
                'rows': rows,
            })
    pair_checks = []
    index = {(row.get('n'), row.get('seed'), row.get('policy')): row for row in rows}
    for n, seed in pairs:
        baseline = index.get((n, seed, POLICIES[0]), {})
        adaptive = index.get((n, seed, POLICIES[1]), {})
        a, b = baseline.get('first_gaussian'), adaptive.get('first_gaussian')
        same = False
        if (isinstance(a, dict) and isinstance(b, dict) and
                all(x.get(key) is not None for x in (a, b) for key in ('center', 'direction'))):
            same = (np.allclose(a.get('center'), b.get('center'), atol=1e-12, rtol=0) and
                    np.allclose(a.get('direction'), b.get('direction'), atol=1e-12, rtol=0))
        pair_checks.append({'n': n, 'seed': seed, 'first_center_direction_match': bool(same)})
    _dump(output_dir / 'summary.json', {
        'status': 'completed_with_censoring' if time.monotonic() >= deadline else 'completed',
        'campaign_wall_seconds': time.monotonic() - started,
        'search_requests_total': sum(row.get('search_requests', 0) for row in rows),
        'fresh_requests_total': sum(row.get('fresh_requests', 0) for row in rows),
        'pair_checks': pair_checks, 'rows': rows,
        'interpretation_limit': 'policy bundle changes both Gaussian width and weight; one escape per arm is not a performance ranking',
    })


def _prepare_payload(output_dir, generator, config, rotation):
    # Execute-mode validation is silent; no calculator is constructed or called.
    if output_dir.exists():
        raise FileExistsError(output_dir)
    qualifications = json.loads(REFERENCE_QUALIFICATION.read_text())
    qualified = {row['n']: bool(row.get('qualified')) for row in qualifications}
    if not all(qualified.get(n, False) for n in SIZES):
        raise ValueError(f'reference qualification missing for requested sizes: {qualified}')
    from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
    inputs = []
    for n in SIZES:
        for seed in SEEDS:
            atoms, child = generator.uniform_volume_cluster(n, seed)
            inputs.append({'n': n, 'seed': seed, 'atom_count': len(atoms),
                'pbc': atoms.pbc.tolist(), 'input_sha256': hashlib.sha256(
                    np.asarray(atoms.positions, dtype='<f8').tobytes()).hexdigest(),
                'search_rng_seed_sequence': {'entropy': child.entropy,
                    'spawn_key': list(child.spawn_key), 'pool_size': child.pool_size}})
    return {
        'output_available': str(output_dir), 'input_source': str(GENERATOR),
        'reference_qualification': str(REFERENCE_QUALIFICATION),
        'sizes': SIZES, 'seeds': SEEDS, 'steps_per_arm': 1,
        'search_request_cap_per_arm_including_initial': ARM_REQUEST_CAP,
        'search_request_cap_total': 8 * ARM_REQUEST_CAP,
        'fresh_request_cap_total': FRESH_REQUEST_CAP_TOTAL,
        'campaign_wall_seconds': CAMPAIGN_WALL_SECONDS,
        'ssw_config': asdict(config), 'recovered_rotation': asdict(rotation),
        'arms': [
            {'name': POLICIES[0], 'gaussian_policy': None},
            {'name': POLICIES[1], 'gaussian_policy': PAMCurvatureGaussian().parameters()},
        ],
        'inputs': inputs, 'PES_requests': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
        help='new directory for all raw inputs, records, and summaries')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true',
        help='qualify inputs/settings and report zero PES calls')
    mode.add_argument('--execute', action='store_true',
        help='run the eight explicitly budgeted one-step trajectories')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output.resolve())
    else:
        execute(args.output.resolve())


if __name__ == '__main__':
    main()
