"""Bounded CPU engineering equivalence; not global-search performance evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


def main():
    from ase.calculators.emt import EMT
    from ase.cluster.icosahedron import Icosahedron
    from ase.io import write
    from pamssw.standalone import ASESurface, SSWConfig, RecoveredRotationSettings, run_ssw
    path = ROOT / 'research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py'
    spec = importlib.util.spec_from_file_location('observer_ledger', path)
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    out = HERE / 'cu13'
    out.mkdir(exist_ok=False)
    atoms = Icosahedron('Cu', 2)
    atoms.positions += np.random.default_rng(25092530).normal(0, .04, (13, 3))
    write(out / 'initial.extxyz', atoms)
    config = SSWConfig(width=.6, rotation_bias=1., max_gaussians=6,
        temperature_K=300., fmax=.03, bias_fmax=.1, relax_steps=1000,
        fd_step=.001, rotation_hvp=39, rotation_tol=.02, forward_force=.1,
        direction_sampling='global', rotation_solver='broyden-euclidean',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total',
        lbfgs_memory=500, rotation_exit_policy='force_or_budget')
    rotation = RecoveredRotationSettings(pre_rotmax=5, rotmax=15,
        pre_ftol=.2, ftol=.02, metric='euclidean', max_force_calls=40)
    total = [0]
    started = time.monotonic()

    class BoundedSurface(ASESurface):
        def evaluate(self, candidate):
            if total[0] >= 40000 or time.monotonic() - started >= 540:
                raise RuntimeError('qualification budget exhausted')
            total[0] += 1
            return super().evaluate(candidate)

    ledger.dump(out / 'protocol.json', dict(config=config, rotation=rotation,
        seed=25092531, git_head=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()))
    rows, baseline = [], None
    for mode in ('legacy', 'compact', 'pause_initial', 'pause_outer'):
        rng = np.random.default_rng(25092531)
        surface = BoundedSurface(EMT())
        before = total[0]
        arm_started = time.monotonic()
        kwargs = dict(steps=4, config=config, rng=rng, recovered_rotation=rotation)
        if mode == 'legacy':
            kwargs['checkpoint_callback'] = lambda cp: False
        else:
            target = {'pause_initial': 0, 'pause_outer': 2}.get(mode)
            kwargs['progress_callback'] = lambda event: event.next_index == target
        result = run_ssw(atoms, surface, **kwargs)
        if mode.startswith('pause_'):
            assert result.status == 'paused', (mode, result.status)
            checkpoint = result.checkpoint
            assert checkpoint.status == 'completed'
            rng = np.random.default_rng(999)
            resumed_surface = BoundedSurface(EMT())
            result = run_ssw(atoms, resumed_surface, steps=4-checkpoint.next_index,
                config=config, rng=rng, recovered_rotation=rotation,
                checkpoint=checkpoint, progress_callback=lambda event: False)
            assert result.evaluation_requests == surface.requests + resumed_surface.requests
        assert result.status == 'completed' and len(result.records) == 4
        state = ledger._jsonable(dict(initial=result.initial, current=result.current,
            best=result.best, minima=result.minima, records=result.records,
            requests=result.evaluation_requests, rng=rng.bit_generator.state))
        ledger.dump(out / f'{mode}-state.json', state)
        if baseline is None:
            baseline = state
        assert state == baseline, f'{mode}: state or cost diverged'
        fresh = ASESurface(EMT())
        energy, forces = fresh.evaluate(result.best)
        row = dict(mode=mode, exact_state_match=True, requests=result.evaluation_requests,
            actual_requests=total[0]-before, fresh_requests=fresh.requests,
            best_energy_eV=energy, best_fmax_eV_A=float(np.linalg.norm(forces, axis=1).max()),
            wall_seconds=time.monotonic()-arm_started)
        rows.append(row)
        ledger.dump(out / 'summary.json', dict(rows=rows, search_requests=total[0],
            complete=len(rows)==4, scope='Engineering equivalence only'))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
