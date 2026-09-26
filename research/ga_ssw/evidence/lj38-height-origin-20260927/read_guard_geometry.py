"""Read saved direct displacements without evaluating a PES."""
import json
from pathlib import Path
import numpy as np


def analyze(source):
    rows = []
    for path in sorted(source.glob('lj38-*/summary.json')):
        data = json.loads(path.read_text())
        for outer in data['outer_steps']:
            for stage in outer['stages']:
                x = np.array(stage['center_A'])
                delta = stage['width_A'] * np.array(stage['mode_direction'])
                y = x + delta
                pairs = np.triu_indices(len(x), 1)
                before = np.linalg.norm(x[:, None] - x[None, :], axis=2)[pairs]
                after = np.linalg.norm(y[:, None] - y[None, :], axis=2)[pairs]
                rows.append(dict(arm=data['arm'], seed=data['seed'], outer=outer['outer_index'],
                    gaussian=stage['gaussian_index'], dmax=float(np.linalg.norm(delta, axis=1).max()),
                    all_pair_min_ratio=float((after/before).min()),
                    min_distance_ratio=float(after.min()/before.min())))
    assert len(rows) == 82
    assert all(r['dmax'] <= 2 and r['min_distance_ratio'] >= .75 for r in rows)
    return rows


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    rows = analyze(root/'cluster-paper-reproduction-20260925/stage-probe-repaired-runs')
    archived = json.loads(Path(__file__).with_name('native-direct-guard-geometry.json').read_text())['rows']
    key = lambda r: (r['arm'], r['seed'], r['outer'], r['gaussian'])
    for a,b in zip(sorted(rows,key=key),sorted(archived,key=key)):
        assert key(a) == key(b)
        for name in ('dmax','all_pair_min_ratio','min_distance_ratio'):
            assert np.isclose(a[name],b[name],rtol=0,atol=1e-14), (name,a,b)
    print('PASS: all 82 archived geometries reproduced, neither default scalar guard triggered; zero PES')
