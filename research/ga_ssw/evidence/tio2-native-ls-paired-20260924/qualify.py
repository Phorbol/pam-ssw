"""Bounded input qualification before the paired periodic LS search."""
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
sys.path.insert(0, str(Path(PLAN['shared_ledger']).parent))
from ledger import CountedSurface, dump, sha256, instrument_calculate
sys.path.insert(0, str(HERE / 'source'))
import numpy as np
from ase.io import read, write
from pamssw.standalone import SSWConfig, run_ssw
from pamssw.standalone.native_ls import (
    initialize_native_ls, TIO_BOND_ENERGIES, TIO_BOND_LENGTHS,
)


def main():
    import torch
    from mace.calculators import MACECalculator
    assert not (HERE / 'qualification.json').exists(), 'preserve existing evidence'
    for rel, digest in PLAN['source_sha256'].items():
        assert sha256(HERE / rel) == digest, rel
    assert sha256(PLAN['shared_ledger']) == PLAN['shared_ledger_sha256']
    assert sha256(PLAN['model']) == PLAN['model_sha256']
    config = SSWConfig(**PLAN['config'])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def calculator():
        return MACECalculator(model_paths=PLAN['model'], head=PLAN['head'],
            device=PLAN['device'], default_dtype=PLAN['dtype'],
            enable_cueq=False, enable_oeq=False)
    calc, fresh = calculator(), calculator()
    counter, fresh_counter = instrument_calculate(calc), instrument_calculate(fresh)
    rows = {}
    for case, digest in PLAN['input_sha256'].items():
        path = HERE / 'inputs' / f'{case}.extxyz'
        assert sha256(path) == digest, case
        original = read(path)
        folder = HERE / ('qualification-' + case)
        folder.mkdir(exist_ok=False)
        calc.reset()
        before = counter['calls']
        surface = CountedSurface(calc, folder / 'requests.jsonl',
            PLAN['qualification_cap_per_case'], PLAN['qualification_wall_per_case'])
        row = {'initial_converged': False, 'fresh_qualified': False}
        try:
            result = run_ssw(original, surface, steps=0, config=config,
                             rng=np.random.default_rng(59))
            dump(folder / 'result.json', result)
            minimum = result.initial
            assert result.evaluation_requests == surface.requests
            row['initial_converged'] = minimum.converged
            work = minimum.atoms.copy()
            write(folder / 'initial.extxyz', work)
            fresh.reset()
            work.calc = fresh
            energy = float(work.get_potential_energy())
            forces = work.get_forces()
            fmax = float(np.linalg.norm(forces, axis=1).max())
            row.update(energy_eV=energy, energy_error_eV=energy-minimum.energy,
                       fmax_eV_A=fmax)
            same = (np.array_equal(work.numbers, original.numbers)
                    and np.array_equal(work.cell.array, original.cell.array)
                    and np.array_equal(work.pbc, original.pbc))
            row['fresh_qualified'] = bool(minimum.converged and same
                and np.isfinite(energy) and np.isfinite(forces).all()
                and fmax <= config.fmax and abs(energy-minimum.energy) <= 1e-6)
            row['LS_geometry'] = {}
            for mode in ('native-mic', 'periodic-images'):
                ls = initialize_native_ls(work, bond_energies=TIO_BOND_ENERGIES,
                    bond_lengths=TIO_BOND_LENGTHS, bond_geometry=mode)
                penalty, penalty_force = ls.potential.evaluate(work)
                row['LS_geometry'][mode] = {
                    'pairs': ls.bond_count, 'penalty_eV': penalty,
                    'penalty_fmax_eV_A': float(np.linalg.norm(penalty_force, axis=1).max()),
                    'finite': bool(np.isfinite(penalty) and np.isfinite(penalty_force).all())}
            distances = work.get_all_distances(mic=True)
            row['minimum_distinct_atom_MIC_distance_A'] = float(distances[np.triu_indices(len(work), 1)].min())
            row['physical_qualification'] = 'force-qualified; geometry requires interpretation; no Hessian or DFT certification'
        except Exception as error:
            row.update(error=repr(error), traceback=traceback.format_exc())
            if hasattr(error, 'result'):
                dump(folder / 'initial-failure.json', error.result)
        row.update(search_requests=surface.requests,
                   calculate_calls=counter['calls']-before, boundary=surface.boundary)
        rows[case] = row
        dump(HERE / 'qualification.json', rows)
    dump(HERE / 'qualification-cost.json', dict(
        search_requests=sum(r['search_requests'] for r in rows.values()),
        fresh_calculate_calls=fresh_counter['calls']))
    return 0 if all(r['initial_converged'] and r['fresh_qualified']
                    and r.get('LS_geometry', {}).get('periodic-images', {}).get('finite')
                    for r in rows.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
