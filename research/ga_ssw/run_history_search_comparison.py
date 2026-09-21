"""Run a bounded fixed-protocol SSW comparison for Safe-LBFGS histories."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from research.ga_ssw.run_recovered_direction_smoke import (
        calculator, decoded_ls, deposited_gaussians,
    )
    from research.ga_ssw.run_public_broyden_ssw import CountedSurface, dump
except ImportError:
    from run_recovered_direction_smoke import calculator, decoded_ls, deposited_gaussians
    from run_public_broyden_ssw import CountedSurface, dump


def _backend_model(plan, override):
    backend = plan['backend']
    kind = backend.get('kind', backend.get('type', 'mace'))
    model = Path(override or backend.get('model', '')) if kind == 'mace' else None
    return kind, model


def _qualification(validation, minima, initial, fmax):
    checks = []
    for index, minimum in enumerate(minima):
        common = dict(index=index,
                      composition_match=bool(np.array_equal(minimum.atoms.numbers, initial.numbers)),
                      pbc_unchanged=bool(np.array_equal(minimum.atoms.pbc, initial.pbc)),
                      fixed_cell=bool(np.array_equal(minimum.atoms.cell.array, initial.cell.array)))
        try:
            energy, forces = validation.evaluate(minimum.atoms)
            force_max = float(np.linalg.norm(forces, axis=1).max())
            common.update(energy=float(energy), energy_error=float(energy - minimum.energy),
                         fmax=force_max,
                         qualified=bool(np.isfinite(energy) and np.isfinite(force_max) and
                                        force_max <= fmax and common['composition_match'] and
                                        common['pbc_unchanged'] and common['fixed_cell']))
        except Exception as error:
            common.update(error=repr(error), qualified=False)
        checks.append(common)
    return checks


def _ls_diagnostics(result, initial, settings):
    from pamssw.standalone.ls_native_reference import NativeLSRuntime
    replay = NativeLSRuntime(initial, settings)
    updates = [record.ls_update for record in result.records if record.ls_update is not None]
    return dict(source='derived initialization replay; zero PES requests',
                initial_bond_count=replay.state.old_bond_count,
                initial_table=replay.state.table,
                initial_pair_count=len(replay.frozen.pairs),
                initial_strengths=list(replay.frozen.strengths),
                initial_strength_sum=float(sum(replay.frozen.strengths)),
                initial_nonzero=bool(replay.frozen.pairs and any(x != 0. for x in replay.frozen.strengths)),
                updates=updates, update_count=len(updates),
                real_response_update=any(
                    'normal_update' in update.get('actions', ()) and
                    update.get('observed_response_mev_per_atom') is not None and
                    np.isfinite(update['observed_response_mev_per_atom'])
                    for update in updates),
                preparations=[record.ls_preparation for record in result.records
                              if record.ls_preparation is not None])


def execute(out, model_override=None):
    plan = json.loads((out / 'plan.json').read_text())
    kind, model = _backend_model(plan, model_override)
    if kind == 'mace' and not model.is_file():
        raise FileNotFoundError(model)
    sys.path.insert(0, str(out / 'source'))
    import pamssw
    if not Path(pamssw.__file__).resolve().is_relative_to(out / 'source'):
        raise RuntimeError('execution must import frozen source package')
    from ase.io import read
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.surface import ASESurface

    base = SSWConfig(**plan['config'])
    recovered = RecoveredDirectionSettings(**plan['recovered'])
    rows = []
    seeds = plan['seeds']
    for case in plan['cases']:
        initial = read(out / 'inputs' / f'{case}.extxyz')
        for seed in seeds:
            for arm in plan['arms']:
                name, memory, use_ls = arm['name'], arm['memory'], bool(arm.get('ls', False))
                folder = out / f'{case}-{name}-seed{seed}'
                folder.mkdir(exist_ok=False)
                ledger = folder / 'requests.jsonl'
                ledger.touch()
                started = time.monotonic()
                search_calc = calculator(kind, model)
                search_init = time.monotonic() - started
                validation_started = time.monotonic()
                validation_calc = calculator(kind, model)
                validation_init = time.monotonic() - validation_started
                surface = CountedSurface(search_calc, ledger,
                                         cap=plan['per_arm_request_cap'],
                                         wall=plan['per_arm_wall_seconds'])
                config = replace(base, lbfgs_memory=memory)
                row = dict(case=case, arm=name, seed=seed, memory=memory, ls=use_ls,
                           backend=kind, execution='started', numerical='unassessed',
                           scientific='fixed-protocol comparison')
                checks = []
                fresh_requests = 0
                settings = None
                try:
                    kwargs = dict(steps=plan['steps'], config=config,
                                  rng=np.random.default_rng(seed),
                                  recovered_direction=recovered)
                    if use_ls:
                        settings = decoded_ls(plan['native_ls'][case])
                        kwargs['ls'] = settings
                    result = run_ssw(initial.copy(), surface, **kwargs)
                    dump(folder / 'result.json', result)
                    validation = ASESurface(validation_calc)
                    checks = _qualification(validation, result.minima, initial, config.fmax)
                    fresh_requests = validation.requests
                    dump(folder / 'qualification.json', checks)
                    if settings is not None:
                        dump(folder / 'native-ls-diagnostics.json',
                             _ls_diagnostics(result, result.initial.atoms, settings))
                    row.update(execution=result.status, minima=len(result.minima),
                               fresh_denominator=len(checks),
                               numerical=('qualified minima' if checks and all(c['qualified'] for c in checks)
                                          else 'failed qualification'),
                               outer_statuses=dict(Counter(r.status for r in result.records)),
                               gaussians=deposited_gaussians(result))
                except Exception as error:
                    row.update(execution='exception', error=repr(error))
                    dump(folder / 'qualification.json', checks)
                row.update(search_requests=surface.requests, fresh_requests=fresh_requests,
                           denials=surface.denials, boundary=surface.boundary,
                           search_initialization_seconds=search_init,
                           validation_initialization_seconds=validation_init,
                           search_seconds=time.monotonic() - surface.started,
                           total_execution_seconds=time.monotonic() - started,
                           checks=checks)
                dump(folder / 'summary.json', row)
                rows.append(row)
                dump(out / 'summary.json', rows)
                print(case, seed, name, row['execution'], surface.requests, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    execute(args.output.resolve(), args.model.resolve() if args.model else None)


if __name__ == '__main__':
    main()
