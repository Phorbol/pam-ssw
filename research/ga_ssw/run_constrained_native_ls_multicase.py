"""Prepare/execute constrained fixed-substrate checkpoint replay.

This runner is deliberately inert unless ``--execute`` is supplied.  It uses
only the public dimer solver and the Cu/Al EMT fixtures from the completed
constrained direction diagnostic; it is a checkpoint/accounting validation,
not a solver comparison.
"""
import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'research/ga_ssw/evidence/constrained-direction-diagnostic-20260912'


def enc(value):
    import numpy as np
    from ase import Atoms
    if isinstance(value, Atoms):
        return dict(numbers=value.numbers.tolist(), positions=value.positions.tolist(),
                    cell=value.cell.array.tolist(), pbc=value.pbc.tolist(),
                    constraints=[enc(c) for c in value.constraints])
    if hasattr(value, 'get_indices'):
        return dict(type=type(value).__name__, indices=np.asarray(value.get_indices(), dtype=int).tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {k: enc(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): enc(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [enc(v) for v in value]
    if hasattr(value, '__dict__'):
        return {str(k): enc(v) for k, v in vars(value).items()
                if k not in ('calculator', 'calc')}
    return value


def dump(path, value):
    path.write_text(json.dumps(enc(value), indent=2, allow_nan=False) + '\n')


def normalized_paid(ledger):
    """Drop surface-local counters while retaining paid geometry/E/F outcomes."""
    return [{key: row[key] for key in ('event', 'charged', 'energy', 'forces', 'atoms')
             if key in row} for row in ledger if row.get('charged')]


def validation_summary(continuous, resumed, *, continuous_requests,
                       split_requests, fresh_continuous, fresh_resumed,
                       config, continuous_rng, resumed_rng):
    """Summarize type-correct checkpoint invariants without doing PES work."""
    cont_current = getattr(getattr(continuous, 'current', None), 'atoms', None)
    resumed_current = getattr(getattr(resumed, 'current', None), 'atoms', None)
    cont_indices = [r['index'] for r in getattr(continuous, 'records', ())
                    if isinstance(r, dict) and 'index' in r]
    resumed_indices = [r['index'] for r in getattr(resumed, 'records', ())
                       if isinstance(r, dict) and 'index' in r]
    final_geometry_equal = (cont_current is not None and resumed_current is not None and
        np.array_equal(cont_current.positions, resumed_current.positions) and
        np.array_equal(cont_current.cell.array, resumed_current.cell.array))
    fresh_qualified = all(c.get('ok') and c.get('active_fmax', np.inf) <= config.fmax and
                          abs(c.get('energy_error', np.inf)) <= 1e-8 and
                          c.get('fixed_cell_exact', False)
                          for c in fresh_continuous + fresh_resumed)
    return dict(expected_three_minima=(len(getattr(continuous, 'minima', ())) == 3 and
                                      len(getattr(resumed, 'minima', ())) == 3),
        fresh_qualified=fresh_qualified,
        continuous_status=getattr(continuous, 'status', None),
        resumed_status=getattr(resumed, 'status', None),
        record_indices_equal=(cont_indices == resumed_indices == [0, 1]),
        final_geometry_equal=bool(final_geometry_equal),
        rng_state_equal=(enc(continuous_rng.bit_generator.state) ==
                         enc(resumed_rng.bit_generator.state)),
        request_accounting=(getattr(continuous, 'requests', None) == continuous_requests),
        resumed_request_accounting=(getattr(resumed, 'requests', None) == split_requests))


def zero_pes_validation_preflight():
    """Exercise result/checkpoint shapes without constructing a calculator."""
    import numpy as np
    from ase import Atoms
    from pamssw.standalone.constrained_reference import (
        ConstrainedSSWConfig, ConstrainedQuenchResult, ConstrainedSSWResult)
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    quench = ConstrainedQuenchResult(atoms.copy(), 0., 0., 0., None,
                                     dict(certified=True), 1)
    result = ConstrainedSSWResult(quench, quench, quench,
        [quench, quench, quench], [dict(index=0), dict(index=1)], 4, 'completed')
    cfg = ConstrainedSSWConfig(width=.2, rotation_bias=100., max_gaussians=14,
        rotation_solver='dimer')
    checks = [dict(ok=True, active_fmax=0., energy_error=0., fixed_cell_exact=True)] * 3
    summary = validation_summary(result, result, continuous_requests=4,
        split_requests=4, fresh_continuous=checks, fresh_resumed=checks,
        config=cfg, continuous_rng=np.random.default_rng(3),
        resumed_rng=np.random.default_rng(3))
    assert all(summary.values()), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), out / 'runner.py')
    snapshot = out / 'source'
    shutil.copytree(ROOT / 'pamssw', snapshot / 'pamssw',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(FIXTURE, out / 'fixture')
    if not args.execute:
        (out / 'PREPARED-NO-PES.md').write_text(
            'Prepared only; no constrained checkpoint API or PES call was run.\n')
        return
    env = os.environ.copy()
    env['PYTHONPATH'] = str(snapshot) + os.pathsep + str(ROOT)
    env['PAM_CONSTRAINED_NATIVE_LS_CHILD'] = '1'
    env['PAM_CONSTRAINED_NATIVE_LS_OUTPUT'] = str(out)
    subprocess.run([sys.executable, str(out / 'runner.py'),
                    '--output', str(out), '--execute'], env=env, check=True)


def _child(out):
    import numpy as np
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms
    import pamssw
    from pamssw.standalone import ASESurface, NativeLSSettings
    from pamssw.standalone.constrained_reference import (
        ConstrainedSSWConfig, run_constrained_ssw, load_constrained_checkpoint)

    source_root = (out / 'source').resolve()
    fixture_root = out / 'fixture'
    imported = Path(pamssw.__file__).resolve()
    if not (imported == source_root or source_root in imported.parents):
        raise RuntimeError(f'child imported non-snapshot pamssw: {imported}')
    dump(out / 'runtime-import.json', dict(pamssw_file=str(imported),
                                          source=str(source_root)))
    zero_pes_validation_preflight()

    def fixture(element):
        data = json.loads((fixture_root / f'{element}-seed3-dimer-result.json').read_text())
        a = Atoms(numbers=data['initial']['numbers'],
                  positions=data['initial']['positions'],
                  cell=data['initial']['cell'], pbc=data['initial']['pbc'])
        fixed = np.asarray(data['fixed_indices'], dtype=int)
        a.set_constraint(FixAtoms(indices=fixed))
        return a, fixed, np.asarray(data['active_indices'], dtype=int), data['config']

    class SharedBudget:
        def __init__(self, cap):
            self.cap = cap
            self.paid = 0
            self.started = None

    def run_case(element):
        atoms, fixed, active, config_data = fixture(element)
        config = ConstrainedSSWConfig(**config_data)
        z = int(atoms.numbers[0])
        length = 2.9 if element == 'Cu' else 3.0
        ls = NativeLSSettings(bond_energies={(z, z): 1.0},
            bond_lengths={(z, z): length}, scale=5.0, target_mev_per_atom=20.0,
            bond_geometry='periodic-images')
        base_pos = atoms.positions.copy()
        base_cell = atoms.cell.array.copy()
        cap = 4000
        continuous_budget = SharedBudget(cap)
        split_budget = SharedBudget(cap)
        wall = 60.0

        class Counted(ASESurface):
            def __init__(self, calculator, ledger_path, budget):
                super().__init__(calculator)
                self.ledger = []
                self.budget = budget
                self.ledger_path = ledger_path
                self.jsonl_path = ledger_path.with_suffix('.jsonl')

            def record(self, row):
                self.ledger.append(row)
                with self.jsonl_path.open('a') as handle:
                    handle.write(json.dumps(enc(row), allow_nan=False) + '\n')

            def evaluate(self, candidate):
                if self.budget.started is None:
                    self.budget.started = time.monotonic()
                if self.requests >= cap:
                    self.record(dict(event='request_cap_denial', charged=False,
                                     requests=self.requests, shared_paid=self.budget.paid))
                    raise RuntimeError('request cap reached')
                if time.monotonic() - self.budget.started >= wall:
                    self.record(dict(event='wall_cap_denial', charged=False,
                                     requests=self.requests, shared_paid=self.budget.paid))
                    raise RuntimeError('wall cap reached')
                if not np.array_equal(candidate.positions[fixed], base_pos[fixed]):
                    self.record(dict(event='fixed_coordinate_violation', charged=False))
                    raise RuntimeError('fixed substrate changed before paid request')
                if not np.array_equal(candidate.cell.array, base_cell):
                    self.record(dict(event='cell_violation', charged=False))
                    raise RuntimeError('cell changed before paid request')
                before = self.requests
                try:
                    if self.budget.paid >= self.budget.cap:
                        self.record(dict(event='shared_request_cap_denial', charged=False,
                                         requests=self.requests, shared_paid=self.budget.paid))
                        raise RuntimeError('shared request cap reached')
                    # ASESurface increments its local request count before
                    # calling the calculator; mirror that paid-failure rule
                    # in the shared ledger before entering the backend.
                    self.budget.paid += 1
                    energy, forces = super().evaluate(candidate)
                except Exception as exc:
                    self.record(dict(event='paid_failure', charged=self.requests > before,
                                     request=self.requests, shared_paid=self.budget.paid,
                                     error=repr(exc), atoms=enc(candidate)))
                    raise
                self.record(dict(event='paid', charged=True, request=self.requests,
                                 shared_paid=self.budget.paid, energy=float(energy),
                                 forces=np.asarray(forces).tolist(), atoms=enc(candidate)))
                return energy, forces

        def fresh(minima):
            checks = []
            for index, minimum in enumerate(minima):
                try:
                    a = minimum.atoms.copy()
                    a.set_constraint()
                    a.calc = EMT()
                    energy = float(a.get_potential_energy())
                    forces = np.asarray(a.get_forces(), dtype=float)
                    checks.append(dict(index=index, ok=True, energy_error=energy-float(minimum.energy),
                        active_fmax=float(np.linalg.norm(forces[active], axis=1).max()),
                        full_raw_fmax=float(np.linalg.norm(forces, axis=1).max()),
                        fixed_cell_exact=bool(np.array_equal(a.positions[fixed], base_pos[fixed]) and
                                              np.array_equal(a.cell.array, base_cell))))
                except Exception as exc:
                    checks.append(dict(index=index, ok=False, error=repr(exc)))
            return checks

        def execute(surface, steps, rng, checkpoint=None, checkpoint_path=None):
            return run_constrained_ssw(atoms,
                surface, steps=steps, config=config, rng=rng, fixed_indices=fixed,
                checkpoint=checkpoint, checkpoint_path=checkpoint_path, ls=ls)

        def native_counts(result):
            cp = getattr(result, 'checkpoint', None)
            state = getattr(cp, 'ls_state', None) if cp is not None else None
            soft = state.get('softening') if isinstance(state, dict) else None
            if soft is None:
                return None
            counts = {'fixed-fixed': 0, 'fixed-mobile': 0, 'mobile-mobile': 0}
            fixed_set = set(map(int, fixed))
            for i, j in soft.pairs:
                fi, fj = int(i) in fixed_set, int(j) in fixed_set
                key = ('fixed-fixed' if fi and fj else
                       'fixed-mobile' if fi != fj else 'mobile-mobile')
                counts[key] += 1
            return dict(counts, total=len(soft.pairs), steps=state.get('steps'),
                        table=state.get('state').table,
                        last_update=state.get('last_update'))

        folder = out / element
        folder.mkdir()
        cont_surface = Counted(EMT(), folder / 'continuous-ledger.json', continuous_budget)
        first_surface = Counted(EMT(), folder / 'first-ledger.json', split_budget)
        resume_surface = Counted(EMT(), folder / 'resumed-ledger.json', split_budget)
        started = time.monotonic()
        continuous = first = resumed = checkpoint = None
        rng_cont = np.random.default_rng(3)
        rng_first = np.random.default_rng(3)
        rng_resume = np.random.default_rng(999)
        error = None
        try:
            continuous = execute(cont_surface, 2, rng_cont,
                                 checkpoint_path=folder / 'continuous.pkl')
            first = execute(first_surface, 1, rng_first,
                            checkpoint_path=folder / 'boundary.pkl')
            checkpoint = load_constrained_checkpoint(folder / 'boundary.pkl')
            resumed = execute(resume_surface, 1, rng_resume,
                              checkpoint=checkpoint, checkpoint_path=folder / 'resumed.pkl')
        except Exception as exc:
            error = repr(exc)

        def result_requests(result):
            if result is None:
                return None
            return getattr(result, 'evaluation_requests',
                           getattr(result, 'requests', None))

        fresh_continuous = fresh(getattr(continuous, 'minima', [])) if continuous else []
        fresh_resumed = fresh(getattr(resumed, 'minima', [])) if resumed else []
        validation = validation_summary(continuous, resumed,
            continuous_requests=cont_surface.requests,
            split_requests=first_surface.requests + resume_surface.requests,
            fresh_continuous=fresh_continuous, fresh_resumed=fresh_resumed,
            config=config, continuous_rng=rng_cont, resumed_rng=rng_resume)
        validation['first_status'] = getattr(first, 'status', None)
        dump(folder / 'validation.json', validation)
        dump(folder / 'continuous-ledger.json', cont_surface.ledger)
        dump(folder / 'first-ledger.json', first_surface.ledger)
        dump(folder / 'resumed-ledger.json', resume_surface.ledger)
        row = dict(element=element,
                   status='completed' if error is None else 'failed', error=error,
                   seed=3, solver='dimer',
                   cap=cap, wall_seconds=wall, continuous_requests=cont_surface.requests,
                   first_requests=first_surface.requests, resumed_requests=resume_surface.requests,
                   shared_paid=split_budget.paid, resumed_total_requests=result_requests(resumed),
                   continuous_result_requests=result_requests(continuous),
                   first_result_requests=result_requests(first),
                   paid_sequence_equal=(normalized_paid(cont_surface.ledger) ==
                                        normalized_paid(first_surface.ledger + resume_surface.ledger)),
                   continuous=enc(continuous), first=enc(first), resumed=enc(resumed),
                   continuous_ledger=cont_surface.ledger,
                   split_ledger=first_surface.ledger + resume_surface.ledger,
                   fresh_continuous=fresh_continuous,
                   fresh_resumed=fresh_resumed, validation=validation,
                   continuous_native=native_counts(continuous),
                   resumed_native=native_counts(resumed),
                   elapsed_seconds=time.monotonic()-started)
        dump(folder / 'result.json', row)
        return row

    # Persist the exact fixture/config contract before any paid call.
    inputs = {}
    configs = {}
    for element in ('Cu', 'Al'):
        fixture_data = json.loads((fixture_root / f'{element}-seed3-dimer-result.json').read_text())
        inputs[element] = fixture_data['initial']
        configs[element] = fixture_data['config']
    dump(out / 'inputs.json', inputs)
    dump(out / 'plan.json', dict(cases=['Cu', 'Al'], seed=3, solver='dimer', steps=2,
        split=[1, 1], request_cap=4000, wall_seconds=60,
        configs=configs, source_fixture=str(FIXTURE),
        purpose='constrained Native-LS checkpoint and paid-request replay; no solver-quality claim'))
    rows = []
    for element in ('Cu', 'Al'):
        try:
            rows.append(run_case(element))
        except Exception as exc:
            rows.append(dict(element=element, status='failed', error=repr(exc)))
        dump(out / 'summary.json', rows)


if __name__ == '__main__':
    if os.environ.get('PAM_CONSTRAINED_NATIVE_LS_CHILD') == '1':
        _child(Path(os.environ['PAM_CONSTRAINED_NATIVE_LS_OUTPUT']))
    else:
        main()
