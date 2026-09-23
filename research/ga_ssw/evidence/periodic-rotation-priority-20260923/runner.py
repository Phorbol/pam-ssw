"""Fixed protocol research runner; frozen independent SSW versus installed ASE BH."""
from pathlib import Path
import argparse
from dataclasses import replace
import json
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
from ledger import CountedSurface, append, dump, instrument_calculate, sha256
from bh_adapter import run_bh
from ase.io import read, write
from pamssw.standalone import SSWConfig, run_ssw, load_ssw_checkpoint
from pamssw.standalone.recovered_rotation import RecoveredRotationSettings


def check_frozen(plan):
    for rel, expected in plan['source_sha256'].items():
        assert sha256(HERE / rel) == expected, rel
    for case, expected in plan['input_sha256'].items():
        assert sha256(HERE / 'inputs' / (case + '.extxyz')) == expected, case
    for rel, expected in plan.get('harness_sha256', {}).items():
        assert sha256(HERE / rel) == expected, rel
    assert sha256(plan['model']) == plan['model_sha256'], 'model changed'
    import ase.optimize.basin
    assert sha256(ase.optimize.basin.__file__) == sha256(HERE / 'ase_basin_reference.py'), 'ASE changed'


def effective_config(plan, case):
    values = dict(plan['ssw_config'])
    values.update(plan['case_config_overrides'][case])
    return SSWConfig(**values)


def validate(minimum, original, config, calc, folder, label):
    row = {'label': label, 'status': 'started'}
    write(folder / (label + '.extxyz'), minimum.atoms)
    append(folder / 'fresh.jsonl', row)
    try:
        calc.reset()
        work = minimum.atoms.copy(); work.calc = calc
        energy = float(work.get_potential_energy())
        forces = np.asarray(work.get_forces(), float)
        fmax = float(np.linalg.norm(forces, axis=1).max())
        cell_ok = bool(np.array_equal(work.cell.array, original.cell.array))
        pbc_ok = bool(np.array_equal(work.pbc, original.pbc))
        numbers_ok = bool(np.array_equal(work.numbers, original.numbers))
        energy_error = energy - minimum.energy
        qualified = bool(np.isfinite(energy) and np.isfinite(forces).all()
                         and fmax <= config.fmax and abs(energy_error) <= 1e-6
                         and cell_ok and pbc_ok and numbers_ok and minimum.converged)
        row.update(status='completed', energy=energy, fmax=fmax, energy_error=energy_error,
                   cell_unchanged=cell_ok, pbc_unchanged=pbc_ok, numbers_unchanged=numbers_ok,
                   numerical_qualified=qualified, physical_qualification='requires_structural_audit')
    except Exception as error:
        row.update(status='error', error=repr(error), numerical_qualified=False)
    append(folder / 'fresh.jsonl', row)
    return row


def run_arm(plan, case, seed, method, atoms, folder, calc, counter, fresh_calc, fresh_counter):
    folder.mkdir(exist_ok=False)
    config = effective_config(plan, case)
    if method == 'recovered':
        config = replace(config, **plan['recovered_config_overrides'])
    dump(folder / 'effective-config.json', config)
    write(folder / 'input.extxyz', atoms)
    calc.reset()
    calls_before, fresh_before = counter['calls'], fresh_counter['calls']
    surface = CountedSurface(calc, folder / 'requests.jsonl', plan['search_cap'], plan['wall_per_arm_seconds'])
    row = dict(case=case, seed=seed, method=method, status='started')
    minima = []
    checkpoint = folder / 'checkpoint.pkl'
    result = None
    def on_quench(value, requests):
        append(folder / 'quenches.jsonl', dict(result=value, requests=requests,
                                              calculate_calls=counter['calls']-calls_before))
        if value.converged:
            minima.append(value)
    try:
        if method in ('ritz', 'recovered'):
            result = run_ssw(atoms.copy(), surface, steps=plan['outer_steps'], config=config,
                             rng=np.random.default_rng(seed), checkpoint_path=checkpoint,
                             recovered_rotation=(RecoveredRotationSettings(**plan['recovered_rotation'])
                                                 if method == 'recovered' else None))
            row.update(status=result.status, steps_completed=len(result.records),
                       accepted=sum(bool(r.accepted) for r in result.records))
            assert result.evaluation_requests == surface.requests
        else:
            with (folder / 'bh.log').open('w') as logfile:
                basin, _ = run_bh(atoms.copy(), surface, seed=seed, config=config,
                                  steps=plan['outer_steps'], on_quench=on_quench, logfile=logfile)
            row.update(status='completed', ase_best_energy=float(basin.Emin))
    except Exception as error:
        row.update(status='exception', error=repr(error), traceback=traceback.format_exc())
        if hasattr(error, 'result'):
            dump(folder / 'initial-failure.json', error.result)
        if method in ('ritz', 'recovered') and checkpoint.exists():
            try:
                result = load_ssw_checkpoint(checkpoint)
                row['recovered_checkpoint'] = True
            except Exception as cp_error:
                row['checkpoint_error'] = repr(cp_error)
    row['search_seconds'] = time.monotonic() - surface.started
    if result is not None:
        dump(folder / 'result.json', result)
        minima = list(result.minima)
    if minima:
        write(folder / 'minima.extxyz', [m.atoms for m in minima])
    checks = []
    if minima:
        # Both archives include all converged proposals, including MC rejections.
        best = min(minima, key=lambda value: value.energy)
        for label, minimum in [('initial', minima[0]), ('best', best)]:
            checks.append(validate(minimum, atoms, config, fresh_calc, folder, label))
    row.update(search_requests=surface.requests, search_calculator_calls=counter['calls']-calls_before,
               denials=surface.denials, boundary=surface.boundary, minima=len(minima),
               fresh_requests=len(checks), fresh_calculator_calls=fresh_counter['calls']-fresh_before,
               fresh=checks)
    if surface.boundary:
        row['execution_class'] = 'budget_censored'
    else:
        row['execution_class'] = row['status']
    dump(folder / 'summary.json', row)
    return row


def execute():
    import torch
    from mace.calculators import MACECalculator
    plan = json.loads((HERE / 'plan.json').read_text())
    check_frozen(plan)
    for case in plan['input_sources']:
        cfg = effective_config(plan, case)
        replace(cfg, **plan['recovered_config_overrides'])
    RecoveredRotationSettings(**plan['recovered_rotation'])
    torch.set_num_threads(1); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    def make():
        return MACECalculator(model_paths=plan['model'], head=plan['head'], device=plan['device'],
                              default_dtype=plan['dtype'], enable_cueq=False, enable_oeq=False)
    calc, fresh_calc = make(), make()
    counter, fresh_counter = instrument_calculate(calc), instrument_calculate(fresh_calc)
    rows = []
    for case in plan['input_sources']:
        atoms = read(HERE / 'inputs' / (case + '.extxyz'))
        for seed in plan['seeds']:
            for method in plan['methods']:
                folder = HERE / f'{case}-{method}-seed{seed}'
                row = run_arm(plan, case, seed, method, atoms, folder, calc, counter, fresh_calc, fresh_counter)
                rows.append(row)
                dump(HERE / 'summary.json', rows)
    assert sum(r['search_requests'] for r in rows) <= plan['search_total_cap']
    assert sum(r['fresh_requests'] for r in rows) <= plan['fresh_total_cap']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print('Prepared only. Use --execute after preflight/review.')
