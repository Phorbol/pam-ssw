"""Compare observer-only replay with frozen histories and paired search costs."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lj_common_cost', HERE/'compare_lj_pilots.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float)) and
                math.isclose(a, b, rel_tol=0, abs_tol=1e-12))
    return a == b


def main():
    roots = dict(global_old=HERE/'runs', paper_old=HERE/'paper-direction-runs',
                 global_new=HERE/'compact-global-runs', paper_new=HERE/'compact-paper-runs')
    arms = {name: comparison.load(root) for name, root in roots.items()}
    checks, pairs = [], []
    for direction in ('global', 'paper'):
        old, new = arms[direction+'_old'], arms[direction+'_new']
        for key in sorted(k for k in new if k[0] == 38):
            _, old_steps, _ = old[key]
            _, new_steps, _ = new[key]
            oi = {r['step']: r for r in old_steps if isinstance(r['step'], int)}
            ni = {r['step']: r for r in new_steps if isinstance(r['step'], int)}
            shared = sorted(oi.keys() & ni.keys())
            mismatches = [index for index in shared if not close(oi[index], ni[index])]
            folder = f'lj{key[0]}-seed{key[1]}'
            identical = np.array_equal(
                read(roots[direction+'_old']/folder/'initial.extxyz').positions,
                read(roots[direction+'_new']/folder/'initial.extxyz').positions)
            checks.append(dict(direction=direction, n=key[0], seed=key[1],
                identical_initial_coordinates=bool(identical),
                shared_boundaries=len(shared), mismatch_indices=mismatches,
                previous_boundaries=len(oi), replay_boundaries=len(ni),
                qualified=bool(identical and shared and not mismatches)))
    for key in sorted(arms['global_new'].keys() & arms['paper_new'].keys()):
        ra, sa, da = arms['global_new'][key]
        rb, sb, db = arms['paper_new'][key]
        budget = min(ra['search_requests'], rb['search_requests'])
        pairs.append(dict(n=key[0], seed=key[1], common_search_budget=budget,
            global_arm=da, paper_arm=db,
            global_best_at_common_budget=comparison.best_at(sa, budget),
            paper_best_at_common_budget=comparison.best_at(sb, budget)))
    output = dict(scope='Reused-input development, not independent success-rate validation. '
                  'Targets require separate geometry-analysis and fresh E/F qualification.',
                  observer_prefix_checks=checks, pairs=pairs,
                  all_observer_checks_pass=bool(checks and all(c['qualified'] for c in checks)))
    (HERE/'compact-comparison.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))
    if not output['all_observer_checks_pass']:
        raise SystemExit('Observer equivalence not qualified; do not rank algorithms.')


if __name__ == '__main__':
    main()
