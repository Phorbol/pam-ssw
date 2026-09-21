"""Offline reconstruction against archived native instructions; no PES calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
from research.ga_ssw.broyden_state_reconstruction import BroydenState


def error(actual, expected):
    delta = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    return dict(absolute=delta, relative_to_max=delta / max(scale, 1e-300), reference_scale=scale)


def main():
    folder = Path('research/ga_ssw/evidence/native-broyden-full-probes-20260912')
    results = []
    for name in ('n6-seed11.json', 'n9-seed29-spectral.json', 'n6-seed29-quartic.json'):
        path = folder / name
        data = json.loads(path.read_text())
        state = BroydenState(data['steps'][0]['g0'], weight=1000.,
                             metric='native_block_sum', history_limit=50, spectral_limit=1e7)
        rows = []
        previous_m = 0
        for row in data['steps']:
            update = state.step(row['x_before'], row['force'])
            m = row['iteration'] - 1
            expected_drop = previous_m + 1 - m if row['step'] else 0
            assert state.history_size == m
            assert update.dropped == expected_drop
            arrays = row['arrays']
            metrics = {key: error(getattr(state, key), np.asarray(arrays[key])[:, :m])
                       for key in ('df', 'u', 'z')}
            metrics['x'] = error(update.x, np.asarray(row['x_after']))
            rows.append(dict(step=row['step'], history_size=m, dropped=update.dropped,
                             spectral_max=update.spectral_max, errors=metrics,
                             inverse_residual=update.action.residual))
            previous_m = m
        closed = BroydenState(data['steps'][0]['g0'], weight=1000.,
                              metric='native_block_sum', history_limit=50, spectral_limit=1e7)
        x = np.asarray(data['steps'][0]['x_before'])
        h = np.asarray(data['hessian'])
        closed_rows = []
        for row in data['steps']:
            force = -h @ x - data.get('quartic', 0.) * x**3
            update = closed.step(x, force)
            x = update.x
            assert closed.history_size == row['iteration'] - 1
            closed_rows.append(dict(step=row['step'], dropped=update.dropped,
                                    history_size=closed.history_size,
                                    x_error=error(x, np.asarray(row['x_after']))))
        results.append(dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            rows=rows, independent_coordinate_force_replay=closed_rows))
    output = dict(protocol='Native x/F replay; Python history carried independently across calls',
                  boundary='Original BRZERO4 arithmetic with SciPy DGGEV substitution; raw nonrotation flags; no PES or efficiency evidence',
                  inputs=results,
                  implementation_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path('research/ga_ssw/broyden_state_reconstruction.py'),
                      Path('research/ga_ssw/broyden_history_reconstruction.py'))})
    target = folder / 'root-state-replay.json'
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(target)
    for result in results:
        print(Path(result['source']).name, 'max_x_error', max(r['errors']['x']['absolute'] for r in result['rows']),
              'drops', [r['dropped'] for r in result['rows']])


if __name__ == '__main__':
    main()
