"""Audit Cu13 direction-only reference replay under an explicit import prefix."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import ase
from ase.calculators.emt import EMT
from ase.io import read
import numpy as np

import pamssw.standalone as standalone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent / 'evidence' / 'cu13-direction-only'
    rows = []
    for solver in ('ritz', 'dimer'):
        reference = json.loads((base / f'3-{solver}.json').read_text())
        config = replace(standalone.SSWConfig(**reference['config']),
                         cluster_frame='direction_only')
        result = standalone.run_ssw(
            read(base / 'initial.extxyz'), standalone.ASESurface(EMT()),
            steps=1, config=config, rng=np.random.default_rng(3))
        actual = result.records[0]
        expected = reference['result']['records'][0]
        rows.append({
            'solver': solver,
            'status': actual.status,
            'evaluation_requests': actual.evaluation_requests,
            'climb_count': len(actual.climb),
            'direction_max_abs_diff': float(np.max(np.abs(
                actual.climb[0]['direction'] -
                np.asarray(expected['climb'][0]['direction'])))),
            'last_position_max_abs_diff_A': float(np.max(np.abs(
                actual.last_atoms.positions -
                np.asarray(expected['last_atoms']['positions'])))),
        })
    payload = {
        'label': args.label,
        'numpy_version': np.__version__,
        'ase_version': ase.__version__,
        'standalone_import': str(Path(standalone.__file__).resolve()),
        'paper_reference_import': str(Path(__import__(
            'pamssw.standalone.paper_reference', fromlist=['x']).__file__).resolve()),
        'rows': rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, separators=(',', ':')))


if __name__ == '__main__':
    main()
