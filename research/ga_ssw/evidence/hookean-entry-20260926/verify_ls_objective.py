"""Independent ASE recomputation of stored LS prequench base energies."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import dict2constraint

HERE = Path(__file__).resolve().parent
rows = []
out = HERE / 'ls-objective-check.json'
if out.exists():
    raise FileExistsError(out)
try:
    for mode in ('paper-v2', 'native', 'native-default'):
        folder = HERE / f'runs-ls-{mode}'
        protocol = json.loads((folder / 'protocol.json').read_text())
        for path in sorted(folder.glob('*-result.json')):
            result = json.loads(path.read_text())
            for record in result['records']:
                prepared = record['ls_preparation']
                atoms = Atoms(**prepared['soft_quench']['atoms'])
                atoms.set_constraint([dict2constraint(json.loads(s)) for s in protocol['specs']])
                atoms.calc = EMT()
                augmented = atoms.get_potential_energy()
                bare = atoms.get_potential_energy(apply_constraint=False)
                error = augmented - prepared['true_energy_after']
                response_error = record['energy_response'] - (
                    prepared['true_energy_after'] - prepared['true_energy_before']) / len(atoms)
                row = dict(mode=mode, run=path.stem, step=record['index'],
                    augmented=augmented, bare=bare, restraint_energy=augmented-bare,
                    stored_energy_error=error, response_error=response_error,
                    prequench_steps=prepared['soft_quench']['optimizer_steps'],
                    response=record['energy_response'])
                rows.append(row)
                assert len(rows) <= 24
                assert abs(error) < 1e-10 and abs(response_error) < 1e-12
                if mode != 'paper-v2':
                    update = record['ls_update']
                    assert abs(update['observed_response_mev_per_atom'] - 1000*record['energy_response']) < 1e-10
    assert len(rows) == 24
    payload = dict(complete=True, fresh_calculations=len(rows), rows=rows)
except Exception as error:
    payload = dict(complete=False, fresh_calculations=len(rows), error=repr(error), rows=rows)
    out.write_text(json.dumps(payload, indent=2) + '\n')
    raise
out.write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(dict(complete=True, fresh_calculations=len(rows),
    max_energy_error=max(abs(r['stored_energy_error']) for r in rows),
    nonzero_restraint_rows=sum(int(r['restraint_energy'] > 1e-10) for r in rows),
    moving_rows=sum(int(r['prequench_steps'] > 0) for r in rows))))
