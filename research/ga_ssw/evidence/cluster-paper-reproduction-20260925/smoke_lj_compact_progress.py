#!/usr/bin/env python3
"""Bounded engineering smoke for compact SSW observation; not search evidence."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from ase import Atoms

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / 'run_lj_pilot.py'
SLM_INPUT = Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/lj38-source-20260912/optim-finish')
OUTPUT = HERE / 'compact-observer-smoke'
LEGACY_LJ38_LOG = HERE / 'runner-smoke/lj38-seed917/outer-steps.jsonl'


def main():
    if not SLM_INPUT.is_file():
        raise FileNotFoundError(SLM_INPUT)
    if not LEGACY_LJ38_LOG.is_file():
        raise FileNotFoundError(LEGACY_LJ38_LOG)
    if OUTPUT.exists():
        raise FileExistsError(f'refusing to overwrite smoke output: {OUTPUT}')
    spec = importlib.util.spec_from_file_location('lj_pilot_compact_smoke', RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.preflight(OUTPUT)
    OUTPUT.mkdir()

    original_surface_init = module.BoundedSurface.__init__

    def bounded_surface_init(self, calculator, *, arm_cap, arm_deadline, total_deadline):
        original_surface_init(self, calculator, arm_cap=min(arm_cap, 1000),
            arm_deadline=arm_deadline, total_deadline=total_deadline)

    module.BoundedSurface.__init__ = bounded_surface_init
    original_generator = module.uniform_volume_cluster
    sources = {
        55: HERE / 'references/lj55.points',
        38: SLM_INPUT,
    }

    def fixture_input(n, seed):
        _, search_seed = original_generator(n, seed)
        positions = np.loadtxt(sources[n]) * module.SIGMA_A
        return Atoms(f'Ar{n}', positions=positions), search_seed

    module.uniform_volume_cluster = fixture_input
    config, rotation = module.make_settings()
    ledger = module.load_ledger()
    deadline = time.monotonic() + 120
    fresh_total = [0]
    rows = []
    for n in (55, 38):
        run_folder = OUTPUT / f'lj{n}-seed917'
        run_folder.mkdir()
        row = module.run_one(n, 917, OUTPUT, config, rotation, deadline,
            fresh_total, ledger, compact_observer=True)
        rows.append(row)
        assert row['observer_mode'] == 'compact'
        assert row['search_requests'] <= 1000
        steps = [json.loads(line) for line in
                 (run_folder / 'outer-steps.jsonl').read_text().splitlines()]
        assert sum(item['step'] == -1 for item in steps) == 1
        if n == 55:
            assert row['status'] == 'first_hit' and row['fresh_checks']
        else:
            compact_step = next((item for item in steps if item['step'] >= 0), None)
            assert compact_step is not None, steps
            assert compact_step['new_minima'], compact_step
            legacy_steps = [json.loads(line) for line in LEGACY_LJ38_LOG.read_text().splitlines()]
            legacy_step = next(item for item in legacy_steps if item['step'] == compact_step['step'])
            for key in ('status', 'accepted', 'step_requests', 'cumulative_requests'):
                assert compact_step[key] == legacy_step[key], (key, compact_step, legacy_step)
            for key in ('landing_energy_eV', 'current_energy_eV', 'best_energy_eV'):
                assert math.isclose(compact_step[key], legacy_step[key], rel_tol=0, abs_tol=1e-12), (
                    key, compact_step[key], legacy_step[key])
            assert math.isclose(compact_step['new_minima'][0]['energy_eV'],
                legacy_step['new_minima'][0]['energy_eV'], rel_tol=0, abs_tol=1e-12)
            assert len(list((run_folder / 'minima').glob('minimum-*.extxyz'))) >= 2
    report = {'status': 'compact_observer_smoke_passed',
        'output': str(OUTPUT), 'search_requests_total': sum(r['search_requests'] for r in rows),
        'fresh_requests_total': fresh_total[0],
        'scope': 'reference LJ55 plus known LJ38 SLM; no random-search evidence',
        'trajectories': [{'n': row['n'], 'status': row['status'],
            'search_requests': row['search_requests'],
            'completed_outer_attempts': row.get('completed_outer_attempts', 0),
            'callback_function_seconds': row['callback_function_seconds']}
            for row in rows]}
    (OUTPUT / 'smoke-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
