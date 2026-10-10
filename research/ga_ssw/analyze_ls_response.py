"""Same-mode decomposition of frozen LS response diagnostics; no oracle calls."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(path):
    result = json.loads((path / 'result.json').read_text())
    report = {key: result.get(key) for key in ('status', 'error', 'requests', 'actual_calculations', 'linear_response', 'baseline')}
    report['source'] = str(path)
    report['rows'] = []
    if not (path / 'baseline.npz').exists():
        return report
    data = np.load(path / 'baseline.npz')
    h0, q = data['H'], data['Q']
    k0 = q.T @ (data['radial'] + data['transverse']) @ q
    l0 = np.linalg.eigvalsh(h0)[0]
    for row in result['rows']:
        a = row['amplitude']
        matrices = np.load(path / f'hessian-{a}.npz')
        hv, k, total = (matrices[key] for key in ('physical', 'bias', 'total'))
        eigen, modes = np.linalg.eigh(total)
        u = modes[:, 0]
        parts = {
            'mode_rotation': float(u @ h0 @ u - l0),
            'direct_at_base': float(a * u @ k0 @ u),
            'physical_geometry': float(u @ (hv-h0) @ u),
            'bias_geometry': float(a * u @ (k-k0) @ u),
        }
        change = float(eigen[0]-l0)
        closure = float(sum(parts.values())-change)
        if abs(closure) > 1e-9 * max(1., np.linalg.norm(total, 2)):
            raise ValueError('same-mode decomposition failed to close')
        predicted = row['predicted_response_per_atom']
        report['rows'].append({
            'amplitude': a, 'eigenvalue_change': change, 'parts': parts, 'closure_error': closure,
            'P_per_atom_eV': row['response_per_atom'],
            'P_over_prediction': row['response_per_atom']/predicted,
            'linear_displacement_relative_error': row['linear_displacement_relative_error'],
            'radius_ratio': row['radius_ratio'], 'resolved_stable': row['resolved_stable'],
            'release': row.get('release'),
        })
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('cases', type=Path, nargs='+')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('analysis output must be new')
    rows = [analyze(path) for path in args.cases]
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
