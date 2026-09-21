"""Bounded raw-VMB2 seed-quantization sweep; no LASP main/PES."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_vmb2 import run


def main(output):
    _blob, segments = load_elf(ELF_DEFAULT)
    cases = []
    for n in (8, 13):
        for k in range(20):
            u = (k + 0.25) / 20.0
            result = run(segments, n, 300.0, np.ones((n, 3), dtype=np.int32),
                         np.zeros((n, 3), dtype=np.float64), u)
            cases.append(result)
    summary = {}
    for n in (8, 13):
        rows = [x for x in cases if x['n'] == n]
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row['first_ran3_seed']].append(np.asarray(row['native']))
        exact = sum(len({a.tobytes() for a in values}) == 1
                    for values in by_seed.values())
        allclose = sum(all(np.allclose(values[0], a, rtol=0, atol=1e-14)
                           for a in values[1:]) for values in by_seed.values())
        summary[str(n)] = {
            'cases': len(rows),
            'unique_first_ran3_seed': len(by_seed),
            'unique_raw_vector_exact': len({a.tobytes() for r in rows for a in [np.asarray(r['native'])]}),
            'same_seed_groups': len(by_seed),
            'same_seed_vector_identical_groups': exact,
            'same_seed_vector_allclose_groups_atol_1e-14': allclose,
            'seed_groups': {str(seed): len(values) for seed, values in by_seed.items()},
            'all_passed': all(r['passed'] for r in rows),
        }
    report = {
        'scope': 'raw VMB2 only; N=8/13, u=(k+.25)/20, k=0..19, ones mask, zero initial, T=300',
        'entry': '0x58c930',
        'native_ran3': True,
        'host_math_hooks': 'cos/log only, as in probe_native_vmb2',
        'pes_requests': 0,
        'summary': summary,
        'cases': cases,
    }
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.output)
