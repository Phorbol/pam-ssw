"""Export owned result pickles without PES; validate exact request ledger and topology."""
import json
import pickle
from pathlib import Path
from ase.io import write

base=Path(__file__).parent
rows=[]
for generation in ('prepared-v1','prepared-v2'):
    for result_path in (base/generation).glob('*/result.pkl'):
        with result_path.open('rb') as stream:
            result=pickle.load(stream)
        original=result.initial.atoms
        ledger=result.initial.evaluation_requests+sum(record.evaluation_requests for record in result.records)
        assert ledger==result.evaluation_requests
        frames=[]
        for index,minimum in enumerate(result.minima):
            atoms=minimum.atoms.copy()
            assert (atoms.numbers==original.numbers).all()
            assert (atoms.cell.array==original.cell.array).all()
            assert (atoms.pbc==original.pbc).all()
            atoms.calc=None
            atoms.info.update(observation_index=index, true_energy_eV=minimum.energy,
                              max_force_eV_A=minimum.max_force, converged=minimum.converged)
            frames.append(atoms)
        write(result_path.parent/'minima.extxyz',frames)
        write(result_path.parent/'current.extxyz',result.current)
        rows.append(dict(path=str(result_path),observations=len(frames),requests=ledger,
                         ledger_closed=True,all_observation_topologies_preserved=True))
(base/'export-checks.json').write_text(json.dumps(dict(rows=rows,pes_calls=0),indent=2)+'\n')
