#!/usr/bin/env python3
"""Zero-PES preflight for the held-out four-arm package."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
LONG = Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-recovered-rotation-long-20260921')
BASELINE = Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-mh1-python-20260919/plan.json')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    baseline = json.loads(BASELINE.read_text())
    report = {'status': 'started', 'checks': {}}
    assert set(plan['cases']) == {'c60_17095', 'c60_17096'}
    assert plan['seed_audit']['observed_matches'] == []
    report['checks']['pre_registered_seeds'] = True
    for case in plan['cases']:
        path = HERE / 'inputs' / f'{case}.traj'
        assert path.exists(), f'missing zero-PES input {path}'
        atoms = read(path)
        assert len(atoms) == 60 and set(atoms.numbers) == {6}
        assert np.array_equal(atoms.pbc, [False, False, False])
        assert np.array_equal(atoms.cell.array, np.diag([50.0, 50.0, 50.0]))
        assert np.all(atoms.positions >= 20.0) and np.all(atoms.positions <= 30.0)
    report['checks']['inputs'] = True
    common = plan['common_ssw_config']
    for key, value in common.items():
        if key == 'rotation_solver':
            continue
        assert value == baseline['config'][key], key
    assert plan['native_mc'] == baseline['native_mc']
    assert plan['arms']['baseline_broyden']['rotation_solver'] == baseline['config']['rotation_solver']
    assert plan['arms']['baseline_broyden']['recovered_rotation'] is None
    assert plan['arms']['recovered_rotation']['recovered_rotation'] == {
        'pre_rotmax': 5, 'rotmax': 15, 'pre_ftol': 0.2, 'ftol': 0.02,
        'metric': 'euclidean', 'max_force_calls': 40}
    report['checks']['arm_difference_only_rotation_policy'] = True
    assert sha256(Path(plan['model'])) == plan['model_sha256']
    report['checks']['model_hash'] = plan['model_sha256']
    for rel, expected in plan['source_sha256'].items():
        assert sha256(HERE / rel) == expected, rel
    for rel, expected in plan['harness_sha256'].items():
        assert sha256(HERE / rel) == expected, rel
    report['checks']['source_and_harness_hashes'] = True
    report['status'] = 'passed'
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    output = HERE / f'preflight-{job_id}.json'
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
