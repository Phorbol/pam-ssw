"""Independent zero-PES ledger/certificate audit of AlOH optimizer arms."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = []
    for folder in sorted(args.directory.glob('frame*')):
        if not folder.is_dir() or not (folder/'summary.json').exists():
            continue
        summary = json.loads((folder/'summary.json').read_text())
        ledger = json.loads((folder/'ledger.json').read_text()) if (folder/'ledger.json').exists() else []
        paid = [x for x in ledger if x.get('charged')]
        assert [x['after'] for x in paid] == list(range(1, len(paid)+1)), folder
        assert len(paid) == summary.get('search_requests', 0), folder
        # These can include dimer displacements and line-search trials. They
        # are diagnostic only, never admitted to the landing timeline.
        small_force_trials = [x for x in paid if np.isfinite(x.get('fmax', float('nan')))
                              and x['fmax'] <= .03]
        timeline, fresh_checks, statuses = [], [], []
        if (folder/'result.json').exists():
            result = json.loads((folder/'result.json').read_text())
            initial = result['initial']
            cumulative = initial['evaluation_requests']
            if initial['converged']:
                timeline.append(dict(request=cumulative, energy=initial['energy']))
            for record in result['records']:
                cumulative += record['evaluation_requests']
                statuses.append(record['status'])
                landing = record.get('landing')
                if landing and landing['converged']:
                    timeline.append(dict(request=cumulative, energy=landing['energy']))
            assert cumulative == result['evaluation_requests'] == len(paid), (folder, cumulative, len(paid))
            fresh = json.loads((folder/'fresh.json').read_text())
            for check in fresh:
                m = result['minima'][check['index']]
                atoms, base = m['atoms'], initial['atoms']
                geometry = (atoms['numbers'] == base['numbers'] and atoms['pbc'] == base['pbc']
                            and np.array_equal(atoms['cell'], base['cell']))
                error, fmax = check.get('energy_error', float('nan')), check.get('fmax', float('nan'))
                valid = (np.isfinite(error) and abs(error) <= 1e-7 and np.isfinite(fmax)
                         and fmax <= .03 and geometry and 'error' not in check)
                fresh_checks.append(dict(index=check['index'], qualified=bool(valid)))
        rows.append(dict(arm=folder.name, frame=summary['frame'], optimizer=summary['optimizer'],
            status=summary['status'], search_requests=len(paid),
            denials=sum(x.get('kind') == 'denial' for x in ledger),
            paid_failures=sum(x.get('kind') == 'failure' for x in paid),
            small_true_force_trial_count=len(small_force_trials),
            first_small_true_force_trial=({k: small_force_trials[0][k] for k in ('after', 'energy', 'fmax')} if small_force_trials else None),
            record_statuses=statuses, timeline=timeline, fresh_checks=fresh_checks))
    comparisons = []
    for frame in sorted({r['frame'] for r in rows}):
        arms = [r for r in rows if r['frame'] == frame]
        prefix = min(r['search_requests'] for r in arms)
        comparison = dict(frame=frame, common_search_prefix=prefix, arms=[])
        for row in arms:
            observations = [x for x in row['timeline'] if x['request'] <= prefix]
            comparison['arms'].append(dict(optimizer=row['optimizer'], observations=len(observations),
                best_energy_eV=min(x['energy'] for x in observations) if observations else None,
                best_delta_initial_eV=(min(x['energy'] for x in observations)-row['timeline'][0]['energy']) if observations else None))
        comparisons.append(comparison)
    output = args.directory/'root-cost-audit.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(dict(rows=rows, comparisons=comparisons,
        scope='Request ledger and force-qualified observations; not distinct basin certification. No new PES requests.'), indent=2)+'\n')
    print(json.dumps(dict(arms=len(rows), search_requests=sum(r['search_requests'] for r in rows),
        checked=sum(len(r['fresh_checks']) for r in rows), qualified=sum(c['qualified'] for r in rows for c in r['fresh_checks']))))


if __name__ == '__main__':
    main()
