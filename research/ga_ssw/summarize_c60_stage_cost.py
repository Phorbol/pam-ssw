"""Count recorded caller phases, without inferring unlogged terminal work."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for seed in (17093, 17094):
        counts = Counter()
        previous = deque(maxlen=2)
        unknown = []
        path = a.native / f'seed{seed}' / 'lasp.out'
        with path.open() as stream:
            for lineno, line in enumerate(stream, 1):
                if line.startswith('Energy,force'):
                    fields = previous[0].split() if len(previous) == 2 else []
                    separator = previous[-1].strip() if previous else ''
                    if len(fields) != 2 or not separator or set(separator) != {'-'}:
                        unknown.append(lineno)
                    else:
                        counts[' '.join(fields)] += 1
                previous.append(line)
        result_path = a.python / f'c60_{seed}' / 'result.json'
        d = json.loads(result_path.read_text())
        events = [event for r in d['records'] for event in r['climb']]
        rotation = sum(e.get('rotation_force_requests', e.get('force_requests', 0)) for e in events)
        bias = sum(e.get('quench_requests', 0) for e in events)
        total = d['evaluation_requests']
        rows.append(dict(seed=seed, native_log=str(path), python_result=str(result_path),
            native=dict(caller_counts=dict(counts), unknown_line_numbers=unknown,
                        printed_evaluations=sum(counts.values())+len(unknown)),
            python=dict(total_evaluations=total, recorded_rotation_calls=rotation,
                recorded_bias_quench_calls=bias,
                remainder_including_initial_true_quench_and_unrecorded_terminal=total-rotation-bias,
                recorded_gaussian_stages=len(events),
                rotation_stops=dict(Counter(e.get('rotation_stop_reason', 'unrecorded') for e in events)))))
    with a.output.open('x') as stream:
        json.dump(dict(scope='Descriptive phase accounting; caller stages are not distinct basins. Partial terminal work may be outside recorded Python phases.', rows=rows), stream, indent=2)


if __name__ == '__main__':
    main()
