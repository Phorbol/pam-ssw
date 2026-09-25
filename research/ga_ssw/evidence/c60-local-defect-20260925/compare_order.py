"""Descriptive input-order comparison; consumes saved checks, makes no PES calls."""
import json
from pathlib import Path

from analyze_escape import events, qualified
from compare_direction import load_arm, costs, best_including_initial, CUTOFFS

HERE = Path(__file__).resolve().parent


def joint_hits(checks, ceiling):
    return [dict(role=c['role'], search_cost=c['search_cost'],
                 delta_ih_eV=c['delta_ih_eV']) for c in checks
            if c['role'].startswith('landing-') and c['search_cost'] <= ceiling
            and qualified(c) and c['energy_window_met']
            and all(c['graphs'][str(t)]['ih_graph_match'] for t in CUTOFFS)]


def main():
    output = HERE / 'direction-order-probe/comparison.json'
    if output.exists():
        raise FileExistsError(output)
    plans = {label: json.loads((HERE / folder / 'plan.json').read_text())
             for label, folder in [('original', 'direction-probe'),
                                   ('permuted', 'direction-order-probe')]}
    rows = []
    for seed in plans['original']['arms']['seeds']:
        for method in plans['original']['methods']:
            arms = {label: load_arm(Path(plan['output']['runs_dir']), seed, method)
                    for label, plan in plans.items()}
            if any(not a['present'] or not isinstance(a.get('search_requests'), int)
                   for a in arms.values()):
                rows.append(dict(seed=seed, method=method, status='missing_arm', arms=arms))
                continue
            ceiling = min(a['search_requests'] for a in arms.values())
            rows.append(dict(seed=seed, method=method, status='compared',
                common_search_prefix=ceiling, arms={label: dict(
                    status=a['status'], source=a['source'], costs=costs(a),
                    full_events=events(a['checks'], a['search_requests']),
                    full_joint_hits=joint_hits(a['checks'], a['search_requests']),
                    common_events=events(a['checks'], ceiling),
                    common_joint_hits=joint_hits(a['checks'], ceiling),
                    best_including_initial=best_including_initial(a))
                    for label, a in arms.items()}))
    output.write_text(json.dumps(dict(
        scope='One physical C60 defect and one fixed row permutation; two numeric seeds per method. No independent structural replication or causal startup-tie attribution.',
        criteria='Joint hit requires qualified fresh force/convergence/composition/cell/PBC, Ih graph at every frozen cutoff, and reference-energy window on the same landing. Counts include repeat visits.',
        interpretation='Same numeric seeds after permutation do not preserve physical random vectors. Prefixes are retrospective; full search and fresh costs remain charged.',
        comparisons=rows), indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
