"""Recompute endpoint Hessian diagnostics from persisted matrices; zero PES."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(directory):
    rows = []
    for endpoint in (0, 1):
        matrices = [np.load(directory / 'matrices' /
                    f'endpoint-{endpoint}-h-{h:.0e}.npy') for h in (1e-4, 5e-5)]
        if any(m.shape != (519, 519) or not np.isfinite(m).all() for m in matrices):
            raise ValueError('Both full519-dimensional matrices are required')
        symmetric = [(m + m.T) / 2 for m in matrices]
        eig = np.linalg.eigvalsh(symmetric[1])
        rows.append(dict(
            endpoint=endpoint, min_eigenvalue=float(eig[0]),
            max_eigenvalue=float(eig[-1]), condition_number=float(eig[-1] / eig[0]),
            step_difference_operator_norm=float(np.linalg.norm(symmetric[1]-symmetric[0], 2)),
            skew_operator_norm=float(np.linalg.norm((matrices[1]-matrices[1].T)/2, 2)),
            skew_frobenius=float(np.linalg.norm(matrices[1]-matrices[1].T)),
            lowest10=eig[:10].tolist(), finite_columns=[519, 519],
            interpretation='observed finite-difference step sensitivity and nonconservativity diagnostics; not a rigorous bound on exact PES Hessian error or an all-q phonon qualification'))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent /
                        'evidence/xxxii-replicated-hessian-completion')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as f:
        json.dump(analyze(args.directory), f, indent=2)
        f.write('\n')
