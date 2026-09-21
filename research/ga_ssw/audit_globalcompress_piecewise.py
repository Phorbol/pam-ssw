"""Independent piecewise C7/C8/C9 audit; no PES or production defaults."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.ga_ssw.oracle_compress_mode import run


def recovered(x, mask, axis, code):
    active = mask[:, 0] > 0
    com = x[active].mean(axis=0)
    lowest = x[mask[:, axis] > 0, axis].min()
    gap = lowest - com[axis]
    region = 1 if gap < 3 else 2 if 3 < gap < 6 else 3 if gap > 6 else 0
    centered = x-com
    out = np.zeros_like(x)
    gate = active & (x[:, axis] < com[axis]+1)
    for j in range(3):
        if j != axis:
            out[gate, j] = -centered[gate, j]/np.linalg.norm(centered[gate], axis=1)
    if code in (7, 8):
        upper = {1:com[axis]+1, 2:com[axis]-2, 3:com[axis]-3}.get(region, -np.inf)
        gate = active & (x[:, axis] > lowest+.5) & (x[:, axis] < upper)
        out[gate, axis] = 2.
    elif code == 9:
        lo, hi = {1:(0., 2.5), 2:(-2.5, 0.), 3:(-5., -2.5)}.get(region, (0., 0.))
        gate = active & (x[:, axis] > com[axis]+lo) & (x[:, axis] < com[axis]+hi)
        out[gate, axis] = -2.
    else:
        raise ValueError(code)
    return out, region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(917)
    rows = []
    for axis in range(3):
        for kind in ('all', 'atom_mask', 'component_mask'):
            x = rng.normal(size=(9, 3))*4
            mask = np.ones((9, 3), dtype=np.int32)
            if kind == 'atom_mask':
                mask[2:4] = 0
            if kind == 'component_mask' and axis:
                mask[:, axis] = (x[:, axis] > 2).astype(int)
                if not mask[:, axis].any():
                    mask[x[:, axis].argmax(), axis] = 1
            cell = np.eye(3)*20
            cell[axis, axis] = 30
            for code in (7, 8, 9):
                result = run(x, cell=cell.ravel(), mask=mask.ravel(),
                             randoms=((code+.2)/11, .95), cache='paired', pair=(0, 0))
                assert result['status'] == 'ok', result
                expected, region = recovered(x, mask, axis, code)
                actual = np.array(result['output'])
                error = float(np.max(abs(actual-expected)))
                result.update(name=f'{axis}-{kind}-{code}', formula_region=region,
                              formula_max_error=error, formula_output=expected.tolist())
                rows.append(result)
    # Independently control the selected minimum relative to the COM using
    # component masks. Cover both strict comparison boundaries (3 and 6).
    for axis in (1, 2):
        for gap in (2., 3., 4., 6., 7.):
            x = rng.normal(size=(9, 3))*4
            values = np.array([-9., -6., -4., -2., 0., 1., 2., 3., gap])
            values[0] -= values.sum()
            x[:, axis] = values
            mask = np.ones((9, 3), dtype=np.int32)
            mask[:, axis] = 0
            mask[-1, axis] = 1
            cell = np.eye(3)*20
            cell[axis, axis] = 30
            for code in (7, 8, 9):
                result = run(x, cell=cell.ravel(), mask=mask.ravel(),
                             randoms=((code+.2)/11, .95), cache='paired', pair=(0, 0))
                assert result['status'] == 'ok', result
                expected, region = recovered(x, mask, axis, code)
                error = float(np.max(abs(np.array(result['output'])-expected)))
                result.update(name=f'{axis}-gap{gap}-{code}', formula_region=region,
                              formula_max_error=error, formula_output=expected.tolist())
                rows.append(result)
    output = dict(scope='original instruction control-vector oracle; zero PES; geometric formulas only', rows=rows)
    with Path(args.output).open('x') as handle:
        json.dump(output, handle, indent=2)
        handle.write('\n')
    worst = max(row['formula_max_error'] for row in rows)
    print(json.dumps(dict(cases=len(rows), max_error=worst,
                         regions=sorted(set(row['formula_region'] for row in rows)))))
    assert worst < 1e-12


if __name__ == '__main__':
    main()
