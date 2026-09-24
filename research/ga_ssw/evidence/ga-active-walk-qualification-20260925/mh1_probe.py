"""Bounded MH-1 C60 active-walk checkpoint probe; execution is opt-in.

Invoke full, pause, and resume as separate Python processes. This script never
submits work and has no retry path. See mh1-plan.md for the frozen protocol.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / 'research/ga_ssw/evidence'
SOURCE = EVIDENCE / 'population-comparison-20260923'
SEED_FILE = SOURCE / 'c60-seed3.json'
MODEL = Path('/home/gengjianrui/.cache/mace/mace-mh-1.model')
RUNS = HERE / 'mh1-runs'
INPUTS = tuple(SOURCE / f'inputs/c60-{n}.extxyz' for n in (17093, 17094, 17095))
REQUEST_CAP = 12000  # per complete equivalent full or prefix+suffix trajectory
FRESH_CAP = 8


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def serializer():
    path = EVIDENCE / 'periodic-rotation-priority-20260923/ledger.py'
    spec = importlib.util.spec_from_file_location('mh1_probe_ledger', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._jsonable


def strip_checkpoints(value):
    if isinstance(value, dict):
        return {key: strip_checkpoints(item) for key, item in value.items()
                if key != 'checkpoint'}
    if isinstance(value, list):
        return [strip_checkpoints(item) for item in value]
    return value


def inputs_and_plan():
    plan = json.loads(SEED_FILE.read_text())
    if tuple(Path(p).name for p in plan['inputs'][:3]) != tuple(p.name for p in INPUTS):
        raise ValueError('first-three C60 input order differs from frozen seed3 config')
    return plan


def run(mode: str) -> None:
    import numpy as np
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone import (ASESurface, NativeMCSettings, PaperGAConfig,
                                   SSWConfig, run_ga_ssw)
    from pamssw.standalone.ga_checkpoint import GACheckpoint
    from pamssw.standalone.legacy_descriptor import cluster_descriptor

    if mode == 'full':
        out = RUNS / 'full'
    elif mode == 'pause':
        out = RUNS / 'split'
    else:
        out = RUNS / 'resumed'
    git = lambda *args: subprocess.check_output(
        ['git', '-C', str(ROOT), *args], text=True).strip()
    if git('status', '--porcelain', '--', 'pamssw'):
        raise RuntimeError('core worktree must be clean before qualified execution')
    out.mkdir(parents=True, exist_ok=False)
    plan = inputs_and_plan()
    initial = [read(path) for path in INPUTS]
    descriptor = plan['descriptor']
    bonds = {tuple(map(int, row[:2])): float(row[2])
             for row in descriptor['bond_lengths']}
    references = tuple(cluster_descriptor(a.numbers, a.positions, bonds,
                        descriptor['neighbor_range']) for a in initial)

    ga = dict(plan['ga'])
    ga.update(quick_steps=1, generations=0, fine_steps=2, cycles=1)
    config = PaperGAConfig(**ga)
    ssw_config = SSWConfig(**plan['ssw'])
    mc = NativeMCSettings(energy_tol=plan['native_mc']['energy_tol_eV'],
                          maxtrap=plan['native_mc']['maxtrap'])

    torch.set_num_threads(1)
    torch.manual_seed(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calc_kwargs = dict(model_paths=str(MODEL), head=plan['backend']['head'],
                       device=plan['backend']['device'],
                       default_dtype=plan['backend']['dtype'],
                       enable_cueq=False, enable_oeq=False)
    surface = ASESurface(MACECalculator(**calc_kwargs))
    checkpoint = GACheckpoint.load(RUNS / 'split' / 'ga.pkl') if mode == 'resume' else None
    # Deliberately wrong fresh RNG on resume: GA must restore the saved state.
    rng = np.random.default_rng(99173 if mode == 'resume' else plan['seed'])
    captured = []

    def callback(state):
        active = state.active_walk
        if mode == 'pause' and active is not None and active.phase == 'fine':
            state.save(out / 'ga.pkl')
            captured.append(state)
            return True
        return False

    before = checkpoint.evaluation_requests if checkpoint is not None else 0
    result = run_ga_ssw(initial, surface, groups=None, references=references,
        descriptor_bonds=bonds, descriptor_weights=descriptor['weights'],
        neighbor_range=descriptor['neighbor_range'], proposal_bond_limits={},
        config=config, ssw_config=ssw_config, rng=rng,
        max_evaluations=REQUEST_CAP, mc=mc, checkpoint=checkpoint,
        checkpoint_callback=callback if mode == 'pause' else None,
        checkpoint_walk_steps=True)
    # Save full Python result for later offline comparison; trusted local pickle.
    import pickle
    with (out / 'result.pkl').open('wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if mode in ('full', 'resume') and (
            result.status not in ('completed', 'completed_with_failures') or
            result.budget_exhausted):
        raise AssertionError(f'{mode} trajectory did not complete: {result.status}')
    if mode == 'pause' and result.status != 'checkpoint_boundary':
        raise AssertionError(f'pause ended unexpectedly: {result.status}')
    if result.evaluation_requests > REQUEST_CAP:
        raise AssertionError('complete-equivalent request cap exceeded')
    if result.evaluation_requests - before != surface.requests:
        raise AssertionError('checkpoint lineage and actual suffix requests disagree')
    if mode == 'pause':
        if len(captured) != 1 or result.checkpoint is None:
            raise AssertionError('fine active-walk pause was not captured exactly once')
        active = GACheckpoint.load(out / 'ga.pkl').active_walk
        if active is None or active.phase != 'fine' or active.steps != 2:
            raise AssertionError('saved nested active-walk state failed roundtrip')
        nested = active.ssw_checkpoint
        if nested.next_index != 1 or nested.next_index >= active.steps:
            raise AssertionError('nested SSW phase/remaining-step invariant failed')
        if active.walk_start_requests + nested.evaluation_requests != result.evaluation_requests:
            raise AssertionError('nested SSW and preceding GA costs do not equal checkpoint lineage')

    encode = serializer()
    atomic_json(out / 'canonical.json', strip_checkpoints(encode(result)))
    atomic_json(out / 'execution.json', {
        'mode': mode, 'git_head': git('rev-parse', 'HEAD'),
        'pamssw_head_tree': git('rev-parse', 'HEAD:pamssw'),
        'python': sys.version, 'executable': sys.executable,
        'model': str(MODEL),
        'configured_model_sha256': plan['backend']['model_sha256'],
        'seed_config': str(SEED_FILE), 'inputs': [str(p) for p in INPUTS],
        'effective_ga': ga, 'effective_ssw': plan['ssw'],
        'effective_descriptor': descriptor, 'native_mc': {
            'energy_tol': plan['native_mc']['energy_tol_eV'],
            'maxtrap': plan['native_mc']['maxtrap']},
        'request_cap': REQUEST_CAP, 'runtime_search_requests': surface.requests,
    })
    summary = {
        'mode': mode, 'status': result.status,
        'actual_suffix_requests': surface.requests,
        'cumulative_requests': result.evaluation_requests,
        'checkpoint_prefix_requests': before,
        'observation_count': len(result.observations),
        'accepted_observations': sum(bool(x.eligible_for_archive) for x in result.observations),
        'observation_ids_unique': len({x.id for x in result.observations}) == len(result.observations),
        'archive_count': len(result.archive),
        'stages': [{'phase': x.phase, 'status': x.status,
                    'requests': x.evaluation_requests, 'observations': x.observations}
                   for x in result.stages],
        'captured_phase': captured[0].active_walk.phase if captured else None,
    }
    # SSW contract: all converged minima are in minima; only nonconverged
    # record landings are additional GA observations. Compare that count with
    # the committed quick/fine stage ledger, without geometric dedup heuristics.
    expected_walk_observations = sum(
        len(walk.minima) + sum(
            getattr(record, 'landing', None) is not None and
            not getattr(record.landing, 'converged', False)
            for record in walk.records)
        for walk in result.walks)
    stage_walk_observations = sum(
        stage.observations for stage in result.stages
        if stage.phase in ('quick', 'fine'))
    summary['walk_ingestion_count'] = expected_walk_observations
    summary['walk_stage_observation_count'] = stage_walk_observations
    atomic_json(out / 'summary.json', summary)
    if (not summary['observation_ids_unique'] or
            expected_walk_observations != stage_walk_observations):
        raise AssertionError('walk landing count and GA stage ingestion differ')
    print(json.dumps(summary, sort_keys=True))


def compare_and_fresh():
    """Offline after the three processes; fresh physical checks are opt-in here."""
    import numpy as np
    import pickle
    import torch
    from pamssw.standalone import ASESurface
    from mace.calculators import MACECalculator

    full_summary = json.loads((RUNS / 'full/summary.json').read_text())
    resumed_summary = json.loads((RUNS / 'resumed/summary.json').read_text())
    split_summary = json.loads((RUNS / 'split/summary.json').read_text())
    total = (full_summary['actual_suffix_requests'] + split_summary['actual_suffix_requests'] +
             resumed_summary['actual_suffix_requests'])
    if (split_summary['actual_suffix_requests'] + resumed_summary['actual_suffix_requests'] !=
            resumed_summary['cumulative_requests'] or total > 2 * REQUEST_CAP):
        raise AssertionError('prefix+suffix cost invariant/cap failed')
    plan = inputs_and_plan()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calc_kwargs = dict(model_paths=str(MODEL), head=plan['backend']['head'],
                       device=plan['backend']['device'], default_dtype='float64',
                       enable_cueq=False, enable_oeq=False)
    fresh_surface = ASESurface(MACECalculator(**calc_kwargs))
    checks = []
    loaded_results = {}
    for mode in ('full', 'resumed'):
        with (RUNS / mode / 'result.pkl').open('rb') as handle:
            result = pickle.load(handle)
        loaded_results[mode] = result
        if (result.status not in ('completed', 'completed_with_failures') or
                result.budget_exhausted or not result.walks or
                len(result.walks[-1].records) != 2 or
                sum(stage.phase == 'fine' for stage in result.stages) != 1):
            raise AssertionError(f'{mode} fine-walk records/stage contract failed')
        candidates = []
        if result.best is not None:
            candidates.append(('best', result.best))
        candidates.append(('current', result.walks[-1].current))
        for label, atoms in candidates:
            if len(checks) >= FRESH_CAP:
                raise AssertionError('fresh qualification cap exceeded')
            energy, forces = fresh_surface.evaluate(atoms)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            checks.append({'mode': mode, 'candidate': label,
                           'energy_eV': float(energy), 'fmax_eV_A': fmax,
                           'numerical_qualified': bool(fmax <= 0.03)})
    split_result = pickle.load((RUNS / 'split/result.pkl').open('rb'))
    resumed_result = loaded_results['resumed']
    if len(resumed_result.observations) < len(split_result.observations):
        raise AssertionError('resumed observations lost the saved prefix')
    encode = serializer()
    if encode(split_result.observations) != encode(resumed_result.observations[:len(split_result.observations)]):
        raise AssertionError('resumed prefix observations changed')
    # Differences are diagnostics only: CUDA trajectories need not be bitwise equal.
    difference = {
        'cumulative_requests': resumed_summary['cumulative_requests'] - full_summary['cumulative_requests'],
        'observation_count': resumed_summary['observation_count'] - full_summary['observation_count'],
        'accepted_observations': resumed_summary['accepted_observations'] - full_summary['accepted_observations'],
        'archive_count': resumed_summary['archive_count'] - full_summary['archive_count'],
        'full_stages': full_summary['stages'], 'resumed_stages': resumed_summary['stages'],
    }
    atomic_json(RUNS / 'analysis.json', {'trajectory_differences': difference,
        'split_suffix_requests': split_summary['actual_suffix_requests'],
        'resumed_suffix_requests': resumed_summary['actual_suffix_requests'],
        'total_search_requests_full_plus_split': total, 'fresh_requests': len(checks),
        'fresh_checks': checks,
        'trajectory_equality_claimed': False,
        'numerical_qualified': all(row['numerical_qualified'] for row in checks)})
    assert len(checks) == 4 and all(row['numerical_qualified'] for row in checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('full', 'pause', 'resume', 'analyze'))
    args = parser.parse_args()
    if args.mode == 'analyze':
        compare_and_fresh()
    else:
        run(args.mode)


if __name__ == '__main__':
    main()
