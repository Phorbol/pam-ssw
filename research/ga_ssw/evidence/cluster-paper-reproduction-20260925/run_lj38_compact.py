"""Reuse the qualified runner for observer-only LJ38 replay; no new algorithm."""
import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lj38_compact_base', HERE/'run_lj_pilot.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', choices=('global', 'paper'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config, rotation = base.make_settings()
    config = replace(config, direction_sampling=args.direction)
    previous = HERE / ('runs' if args.direction == 'global' else 'paper-direction-runs')
    old_spec = importlib.util.spec_from_file_location('lj38_frozen_previous', previous/'run_lj_pilot.py')
    frozen = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(frozen)
    for seed in base.SEEDS:
        old = previous / f'lj38-seed{seed}'
        summary = json.loads((old/'summary.json').read_text())
        assert summary['settings']['ssw_config'] == asdict(config)
        assert summary['settings']['recovered_rotation'] == asdict(rotation)
        atoms, search_seed = base.uniform_volume_cluster(38, seed)
        old_atoms, old_search_seed = frozen.uniform_volume_cluster(38, seed)
        assert np.array_equal(atoms.positions, old_atoms.positions)
        assert np.array_equal(search_seed.generate_state(4), old_search_seed.generate_state(4))
    print('Preflight: identical starting coordinates and numerical settings; zero PES.', flush=True)
    if not args.execute:
        return
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    ledger = base.load_ledger()
    for name in ('run_lj38_compact.py', 'run_lj_pilot.py', 'lj38-compact-plan.md'):
        shutil.copy2(HERE/name, out/name)
    ledger.dump(out/'execution.json', dict(
        git_head=subprocess.check_output(['git','-C',str(base.ROOT),'rev-parse','HEAD'],text=True).strip(),
        python=sys.executable, previous=str(previous), observer='compact',
        config=config, recovered_rotation=rotation, search_cap=1600000, fresh_cap=4))
    started = time.monotonic()
    deadline = started + 3600
    fresh, rows = [0], []
    for seed in base.SEEDS:
        folder = out / f'lj38-seed{seed}'
        folder.mkdir()
        if time.monotonic() >= deadline:
            row = dict(n=38, seed=seed, status='not_run_campaign_wall_cap',
                       search_requests=0, fresh_requests=0)
            ledger.dump(folder/'summary.json', row)
        else:
            row = base.run_one(38, seed, out, config, rotation, deadline, fresh,
                               ledger, compact_observer=True)
        rows.append(row)
        ledger.dump(out/'summary.json', dict(trajectories=rows,
            search_requests_total=sum(r['search_requests'] for r in rows),
            fresh_requests_total=fresh[0], wall_seconds_total=time.monotonic()-started,
            status='running' if len(rows)<2 else 'complete_or_censored'))


if __name__ == '__main__':
    main()
