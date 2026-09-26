"""Reconstruct saved forward-force heights; no trajectory generation or tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from research.ga_ssw.full_pair_lj import FullPairLJ
from pamssw.standalone.gaussian import ProjectedGaussian


def analyze(source):
    rows, costs = [], []
    requests = 0
    for path in sorted(source.glob('lj38-*/summary.json')):
        data = json.loads(path.read_text())
        for outer in data['outer_steps']:
            history = []
            for stage in outer['stages']:
                center = np.array(stage['center_A'])
                mode = np.array(stage['mode_direction'])
                assert np.isclose(np.linalg.norm(mode), 1.0)
                width = stage['width_A']
                atoms = Atoms('Ar38', positions=center)
                atoms.calc = FullPairLJ()
                e0 = atoms.get_potential_energy()
                f0 = atoms.get_forces()
                requests += 1
                atoms.positions += width * mode
                e1 = atoms.get_potential_energy()
                f1 = atoms.get_forces()
                requests += 1
                physical = float(np.sum(f1 * mode))
                old = sum(float(np.sum(term.evaluate(atoms)[1] * mode)) for term in history)
                target = data['settings']['ssw_config']['forward_force']
                height = (target - physical - old) * width * np.exp(.5)
                pairs = np.triu_indices(len(atoms), 1)
                def minpair(x):
                    return float(np.linalg.norm(x[:, None] - x[None, :], axis=2)[pairs].min())
                rows.append(dict(arm=data['arm'], seed=data['seed'], outer=outer['outer_index'],
                    gaussian=stage['gaussian_index'], height=stage['height_eV'], reconstructed_height=height,
                    height_error=height-stage['height_eV'], center_min_distance_sigma=minpair(center)/2.7,
                    probe_min_distance_sigma=minpair(atoms.positions)/2.7,
                    physical_energy_increase_eV=e1-e0, center_projected_force=float(np.sum(f0*mode)),
                    probe_projected_physical_force=physical, probe_projected_history_force=old,
                    quench_requests=stage['quench_requests'], rotation_requests=stage['rotation_force_cost']))
                history.append(ProjectedGaussian(center, mode, width, stage['height_eV']))
        costs.append(dict(arm=data['arm'],seed=data['seed'],search_requests=data['search_requests'],
                          initial_requests=data['initial_requests']))
    assert len(rows) == 82, 'Unexpected source set; review the protocol before extending it'
    assert requests == 164
    error = max(abs(row['height_error']) for row in rows)
    # Saved JSON has full-precision centers/modes; this only checks reconstruction.
    assert error < 1e-7, error
    return dict(scope='Outcome-selected saved-stage diagnostic, not search performance',
                source=str(source.resolve()), evaluation_requests=requests, max_height_error=error,
                rows=rows, original_costs=costs)


def curvature_readout(result):
    """Compare stored local curvature with the already evaluated finite probe."""
    lookup = {}
    for path in Path(result['source']).glob('lj38-*/summary.json'):
        data = json.loads(path.read_text())
        for outer in data['outer_steps']:
            for stage in outer['stages']:
                lookup[(data['arm'], data['seed'], outer['outer_index'], stage['gaussian_index'])] = stage
    rows = []
    for row in result['rows']:
        identity = {key: row[key] for key in ('arm', 'seed', 'outer', 'gaussian')}
        stage = lookup[tuple(identity.values())]
        trace = [item for item in stage['rotation_trace'] if 'real_curvature' in item][-1]
        curvature = trace['real_curvature']
        prediction = row['center_projected_force'] - curvature * stage['width_A']
        rows.append(dict(identity, height=row['height'], stored_directional_curvature=curvature,
                         linear_predicted_probe_force=prediction,
                         actual_probe_force=row['probe_projected_physical_force'],
                         linear_prediction_error=row['probe_projected_physical_force']-prediction))
    return dict(scope='Reuse recorded finite-difference curvature; no new E/F', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path)
    inputs.add_argument('--from-result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = (curvature_readout(json.loads(args.from_result.read_text()))
              if args.from_result else analyze(args.source))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','original_costs')}))
