"""Offline force/energy gate diagnostic on recorded Python first moves.

No oracle, no counter equivalence assumption, no prediction of continued paths.
Only completed converged stages with a recorded true-energy check are used.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pamssw.standalone.gaussian import ProjectedGaussian


def replay(root):
    output = []
    for case in ('c60_17093', 'c60_17094', 'cu55', 'water15'):
        folder = root / ('c60' if case.startswith('c60') else 'controls') / (case+'-strict')
        result = json.loads((folder/'result.json').read_text())
        initial = result['initial']
        record = result['records'][0]
        ledger = []
        for line in (folder/'requests.jsonl').open():
            row = json.loads(line)
            if row['kind'] == 'search':
                ledger.append(row)
            if len(ledger) >= initial['evaluation_requests']+record['evaluation_requests']:
                break
        end = initial['evaluation_requests']
        terms, stages = [], []
        for event in record['climb']:
            end += event['requests']
            count = event.get('quench_requests', 0)
            if not count or 'true_energy' not in event or event['status'] != 'converged':
                raise ValueError('unsupported incomplete first-move stage')
            term = ProjectedGaussian(np.array(event['center']), np.array(event['direction']),
                                     event['width'], event['weight'])
            terms.append(term)
            requests = ledger[end-1-count:end-1]
            np.testing.assert_allclose(requests[0]['atoms']['positions'],
                                       term.center+term.sigma*term.direction, atol=1e-12, rtol=0)
            trace = []
            for index, row in enumerate(requests, 1):
                atoms = SimpleNamespace(positions=np.array(row['atoms']['positions']))
                energy = row['energy']
                forces = np.array(row['forces'])
                for old in terms:
                    de, df = old.evaluate(atoms)
                    energy += de
                    forces += df
                trace.append(dict(local_evaluation=index, global_request=row['request'],
                                  base_energy=row['energy'], modified_energy=energy,
                                  modified_component=float(np.abs(forces).max()),
                                  modified_atom_norm=float(np.linalg.norm(forces,axis=1).max()),
                                  force_component_gate=bool(np.abs(forces).max()<.15),
                                  initial_lower_energy_gate=bool(row['energy']<initial['energy']-.1)))
            assert abs(trace[-1]['modified_energy']-event['biased_energy']) < 1e-8
            assert abs(trace[-1]['modified_atom_norm']-event['max_force']) < 1e-8
            stages.append(dict(index=event['index'], actual_quench_requests=count,
                               first_force_gate=next((t['local_evaluation'] for t in trace if t['force_component_gate']),None),
                               first_energy_gate=next((t['local_evaluation'] for t in trace if t['initial_lower_energy_gate']),None),
                               trace=trace))
        output.append(dict(case=case,source=str(folder.resolve()),stages=stages))
    return dict(scope='Offline scalar gates on actual Python trial evaluations; no native dispatch/counter equivalence or counterfactual search outcome.',
                force_gate='max(abs(modified_force)) < 0.15 eV/A',
                energy_gate='physical_energy < first_move_initial_energy - 0.1 eV', cases=output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=replay(args.input)
    with args.output.open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write('\n')
