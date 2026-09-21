"""Bounded EMT integration check for GA SSW options and quick-boundary resume.

This is an interface/checkpoint qualification run, not a GA search benchmark.
It deliberately reuses the three qualified Cu13 inputs and GA settings from
``run_ga_checkpoint_emt.py`` while enabling explicit native MC and recovered
CBD rotation settings.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/ga_ssw/evidence/ga-walker-options-emt-20260920"
MAX_EVALUATIONS = 20_000
FRESH_MAX = 100
WALL_SECONDS = 180


class WallTimeout(BaseException):
    pass


def _freeze_before_import():
    if OUT.exists():
        execution_artifacts = ('source', 'manifest.json', 'quick.chk', 'result.json')
        present = [name for name in execution_artifacts if (OUT / name).exists()]
        if present:
            raise FileExistsError(
                f"refusing to overwrite existing execution artifacts in {OUT}: {present}")
    else:
        OUT.mkdir(parents=True)
    source = OUT / "source"
    source.mkdir()
    shutil.copy2(__file__, source / Path(__file__).name)
    shutil.copy2(ROOT / "research/ga_ssw/run_ga_checkpoint_emt.py",
                 source / "run_ga_checkpoint_emt.py")
    shutil.copytree(ROOT / "pamssw", source / "pamssw")
    return source


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if is_dataclass(value):
        return {k: _json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return repr(value)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ga_kwargs(initial, bonds, refs, seed, surface, *, mc, recovered_rotation):
    from pamssw.standalone import PaperGAConfig, SSWConfig
    import numpy as np

    config = PaperGAConfig(
        quick_steps=1, generations=1, generation_steps=1, fine_steps=1,
        ga_candidates=8, regions=1, fine_regions=1, quench_fmax=.01,
        quench_steps=200, proposal_max_batches=2,
        proposal_max_cut_attempts=1000, proposal_max_pair_attempts=10000,
        partition_max_draws=10000, projection_tolerance=1e-4,
        energy_window=10., proposal_type=0,
        proposal_max_insertion_attempts=10000)
    ssw = SSWConfig(
        width=.2, rotation_bias=1., max_gaussians=14,
        temperature_K=300., fmax=.01, relax_steps=200, fd_step=.001,
        rotation_hvp=39, rotation_tol=.02, forward_force=.1,
        direction_sampling='global', rotation_solver='broyden-euclidean',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total',
        lbfgs_memory=500, bias_fmax=.1, rotation_exit_policy='force_or_budget')
    return dict(
        initial=initial, surface=surface, groups=None, references=refs,
        descriptor_bonds=bonds, descriptor_weights=(.3, .2, .2, .1, .1, .1),
        neighbor_range=2., proposal_bond_limits={}, config=config,
        ssw_config=ssw, rng=np.random.default_rng(seed),
        max_evaluations=MAX_EVALUATIONS, mc=mc,
        recovered_rotation=recovered_rotation)


class LedgerSurface:
    def __init__(self):
        from pamssw.standalone import ASESurface
        from ase.calculators.emt import EMT
        self._surface = ASESurface(EMT())
        self.ledger = []

    @property
    def requests(self):
        return self._surface.requests

    def evaluate(self, atoms):
        import numpy as np
        try:
            energy, forces = self._surface.evaluate(atoms)
            self.ledger.append({
                'ok': True, 'positions': atoms.positions.tolist(),
                'energy': float(energy), 'forces': np.asarray(forces).tolist()})
            return energy, forces
        except Exception as error:
            self.ledger.append({
                'ok': False, 'positions': atoms.positions.tolist(),
                'error': f'{type(error).__name__}: {error}'})
            raise


def _observation_fingerprint(result):
    return [
        {
            'phase': obs.phase, 'generation': obs.generation,
            'seed_id': obs.seed_id, 'parent_ids': list(obs.parent_ids),
            'operator': obs.operator,
            'eligible_for_archive': obs.eligible_for_archive,
            'positions': obs.result.atoms.positions.tolist(),
            'energy': float(obs.result.energy),
        }
        for obs in result.observations
    ]


def _ledger_fingerprint(rows):
    return [
        {'ok': row.get('ok'), 'positions': row.get('positions'),
         'energy': row.get('energy'), 'forces': row.get('forces'),
         'error': row.get('error')}
        for row in rows
    ]


def _write_run_snapshot(label, surface, result=None, error=None, elapsed=None):
    payload = {
        'label': label,
        'elapsed_seconds': elapsed,
        'error': None if error is None else {
            'type': type(error).__name__, 'message': str(error)},
        'ledger_requests': surface.requests,
        'ledger': surface.ledger,
    }
    if result is not None:
        payload['state'] = _state_rows(result)
    (OUT / f'{label}-snapshot.json').write_text(
        json.dumps(_json_value(payload), indent=2) + '\n')


def _run_with_wall(label, surface, invoke, seconds=WALL_SECONDS):
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)

    def alarm(_signum, _frame):
        raise WallTimeout(f'{label} exceeded remaining {seconds}s wall limit')

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, seconds))
    try:
        result = invoke()
        _write_run_snapshot(label, surface, result=result,
                            elapsed=time.monotonic() - started)
        return result
    except BaseException as error:
        _write_run_snapshot(label, surface, error=error,
                            elapsed=time.monotonic() - started)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _audit_walk_options(result):
    records = []
    missing_mc = []
    bad_solver = []
    failure_records = []
    eligible_count = 0
    mc_count = 0
    cbd_count = 0
    for walk_index, walk in enumerate(result.walks):
        for record in walk.records:
            climb = list(record.climb)
            landing = record.landing
            eligible_landing = bool(
                landing is not None and getattr(landing, 'surface', None) == 'true'
                and getattr(landing, 'converged', False)
            )
            solver_events = [event.get('rotation_solver') for event in climb
                             if event.get('rotation_solver') is not None]
            recovered_events = [solver for solver in solver_events
                                if solver == 'recovered-cbd']
            if eligible_landing:
                eligible_count += 1
            if record.mc_telemetry is not None:
                mc_count += 1
            cbd_count += len(recovered_events)
            entry = {
                'walk_index': walk_index, 'step': record.index,
                'status': record.status, 'accepted': bool(record.accepted),
                'evaluation_requests': record.evaluation_requests,
                'eligible_landing': eligible_landing,
                'has_mc_telemetry': record.mc_telemetry is not None,
                'climb_count': len(climb),
                'rotation_solvers': [event.get('rotation_solver') for event in climb],
                'climb_statuses': [event.get('status') for event in climb],
            }
            records.append(entry)
            if eligible_landing and record.mc_telemetry is None:
                missing_mc.append(entry)
            if any(solver != 'recovered-cbd' for solver in solver_events):
                bad_solver.append(entry)
            if (not eligible_landing) or record.mc_telemetry is None or not recovered_events:
                failure_records.append(entry)
    if missing_mc or bad_solver:
        raise AssertionError(json.dumps({
            'missing_mc_telemetry_on_eligible_landing': missing_mc,
            'non_recovered_cbd_climb': bad_solver}, indent=2))
    return {
        'records': records,
        'eligible_landing_count': eligible_count,
        'mc_telemetry_count': mc_count,
        'recovered_cbd_event_count': cbd_count,
        'failure_or_unselected_records': failure_records,
    }


def _connected(atoms, cutoff=3.0):
    import numpy as np
    distances = atoms.get_all_distances(mic=False)
    seen = {0}
    pending = [0]
    while pending:
        i = pending.pop()
        for j in range(len(atoms)):
            if j not in seen and distances[i, j] <= cutoff:
                seen.add(j)
                pending.append(j)
    return len(seen) == len(atoms)


def _fresh_archive(result):
    from ase.calculators.emt import EMT
    import numpy as np
    rows = []
    for index, item in enumerate(result.archive[:FRESH_MAX]):
        atoms = item['atoms'].copy()
        atoms.calc = EMT()
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        archive_energy = float(item['energy'])
        max_force = float(np.linalg.norm(forces, axis=1).max())
        rows.append({
            'archive_index': index, 'energy_eV': energy,
            'archive_energy_eV': archive_energy,
            'energy_error_eV': abs(energy - archive_energy),
            'energy_consistent': bool(abs(energy - archive_energy) <= 1e-10),
            'max_force_eV_A': max_force,
            'force_qualified_at_0.01': bool(max_force <= .01),
            'connected_at_3A': bool(_connected(atoms)),
        })
    return {
        'frames_available': len(result.archive),
        'frames_checked': len(rows),
        'frames_unchecked': max(0, len(result.archive) - len(rows)),
        'rows': rows,
        'all_energy_consistent': bool(rows) and all(row['energy_consistent'] for row in rows),
        'all_force_qualified_at_0.01': bool(rows) and all(row['force_qualified_at_0.01'] for row in rows),
        'all_connected_at_3A': bool(rows) and all(row['connected_at_3A'] for row in rows),
    }


def _state_rows(result):
    return {
        'status': result.status,
        'evaluation_requests': result.evaluation_requests,
        'budget_limit': result.budget_limit,
        'budget_exhausted': result.budget_exhausted,
        'observations': len(result.observations), 'archive': len(result.archive),
        'stages': [
            {'phase': stage.phase, 'generation': stage.generation,
             'seed_id': stage.seed_id, 'status': stage.status,
             'evaluation_requests': stage.evaluation_requests,
             'observations': stage.observations, 'cycle': stage.cycle}
            for stage in result.stages
        ],
        'failures': [
            {'phase': failure.phase, 'generation': failure.generation,
             'seed_id': failure.seed_id, 'reason': failure.reason,
             'evaluation_requests': failure.evaluation_requests}
            for failure in result.failures
        ],
        'walks': len(result.walks),
    }


def main():
    source = _freeze_before_import()
    os.chdir(ROOT)
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(ROOT / 'research/ga_ssw'))
    import pamssw
    pamssw_path = Path(pamssw.__file__).resolve()
    try:
        pamssw_path.relative_to(source)
    except ValueError as error:
        raise AssertionError(
            f'expected frozen pamssw import under {source}, got {pamssw_path}') from error
    import numpy as np
    from pamssw.standalone import RecoveredRotationSettings, run_ga_ssw
    from pamssw.standalone.ga_checkpoint import GACheckpoint
    from pamssw.standalone.native_mc import NativeMCSettings
    from run_ga_checkpoint_emt import _inputs

    initial, bonds, refs, input_source, groups = _inputs()
    mc = NativeMCSettings(.1, 99999)
    recovered = RecoveredRotationSettings(
        pre_rotmax=5, rotmax=15, pre_ftol=1., ftol=.1,
        metric='euclidean', max_force_calls=40)
    manifest = {
        'input_source': str(input_source), 'groups': groups, 'seed': 3,
        'max_evaluations_per_trajectory': MAX_EVALUATIONS,
        'fresh_max_frames': FRESH_MAX, 'mc': _json_value(mc),
        'recovered_rotation': _json_value(recovered),
        'wall_seconds_per_trajectory': WALL_SECONDS,
        'pamssw_import_path': str(pamssw_path),
        'runner_sha256': _sha256(Path(__file__)),
        'source_snapshot': str(source),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run_one(label, *, checkpoint=None, stop_quick=False, seconds=WALL_SECONDS):
        surface = LedgerSurface()
        captured = []
        callback = None
        if stop_quick:
            def callback(state):
                captured.append(state)
                return state.phase == 'quick_complete'
        kwargs = _ga_kwargs(initial, bonds, refs, 3, surface,
                            mc=mc, recovered_rotation=recovered)
        if checkpoint is not None:
            kwargs['checkpoint'] = checkpoint
        result = _run_with_wall(
            label, surface,
            lambda: run_ga_ssw(**kwargs, checkpoint_callback=callback), seconds=seconds)
        return result, surface, captured

    full, full_surface, _ = run_one('full')
    split_started = time.monotonic()
    partial, split_surface, captured = run_one('quick', stop_quick=True)
    if partial.checkpoint is None or not captured or captured[-1].phase != 'quick_complete':
        raise AssertionError('quick_complete checkpoint was not produced')
    checkpoint_path = OUT / 'quick.chk'
    partial.checkpoint.save(checkpoint_path)
    resumed, resumed_surface, _ = run_one(
        'resumed', checkpoint=GACheckpoint.load(checkpoint_path),
        seconds=WALL_SECONDS - (time.monotonic() - split_started))

    if split_surface.requests + resumed_surface.requests > MAX_EVALUATIONS:
        raise AssertionError('split plus resume exceeded per-trajectory budget')
    if full.evaluation_requests > MAX_EVALUATIONS:
        raise AssertionError('full trajectory exceeded per-trajectory budget')
    if _observation_fingerprint(full) != _observation_fingerprint(resumed):
        raise AssertionError('full and quick-resume observations differ')
    if _ledger_fingerprint(full_surface.ledger) != _ledger_fingerprint(
            split_surface.ledger + resumed_surface.ledger):
        raise AssertionError('full and quick-resume surface ledger differs')

    full_walk_audit = _audit_walk_options(full)
    resumed_walk_audit = _audit_walk_options(resumed)
    for label, audit in (('full', full_walk_audit), ('quick-resume', resumed_walk_audit)):
        if audit['mc_telemetry_count'] == 0:
            raise AssertionError(f'{label} GA run contained no actual MC telemetry')
        if audit['recovered_cbd_event_count'] == 0:
            raise AssertionError(f'{label} GA run contained no recovered-CBD event')
    result = {
        'protocol': 'GA walker option and quick-boundary checkpoint integration',
        'full': {**_state_rows(full), 'ledger_requests': full_surface.requests},
        'quick_partial': {**_state_rows(partial), 'ledger_requests': split_surface.requests,
                          'checkpoint_phase': partial.checkpoint.phase},
        'quick_resumed': {**_state_rows(resumed), 'ledger_requests': resumed_surface.requests},
        'split_resume_total_requests': split_surface.requests + resumed_surface.requests,
        'full_vs_resume_observations_match': True,
        'full_vs_resume_surface_ledger_match': True,
        'full_walk_audit': full_walk_audit,
        'resumed_walk_audit': resumed_walk_audit,
        'fresh_validation': _fresh_archive(resumed),
        'status_is_integration_check_only': True,
    }
    (OUT / 'result.json').write_text(json.dumps(_json_value(result), indent=2) + '\n')
    print(json.dumps(_json_value(result)))


if __name__ == '__main__':
    main()
