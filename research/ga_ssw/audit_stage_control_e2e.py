"""Offline cost, geometry and matched-prefix audit of the frozen stage probe."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def audit(root):
    manifest = read(root / 'source-manifest.json')
    for name, digest in manifest['source_files'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest, name
    rows = []
    traces = {}
    for folder in sorted(root.glob('*-seed11')):
        summary = read(folder / 'summary.json')
        metadata = read(folder / 'arm-metadata.json')
        result = read(folder / 'raw-result.json')
        fresh = read(folder / 'fresh-checks.json')
        ledger = [json.loads(line) for line in (folder / 'evaluations.jsonl').read_text().splitlines()]
        paid = [x for x in ledger if x['charged']]
        assert len(paid) == summary['requests'] <= 4000
        assert [x['request'] for x in paid] == list(range(1, len(paid) + 1))
        assert all(not x['charged'] for x in ledger if x['kind'] == 'denied')
        for event in ledger:
            for key in ('numbers', 'cell', 'pbc'):
                assert event['atoms'][key] == metadata['input'][key], (folder.name, key)
        cfg = metadata['config']
        assert cfg['rotation_solver'] == 'dimer' and cfg['bias_fmax'] == .1
        assert cfg['fmax'] == .01 and cfg['relax_steps'] == 200
        assert cfg['lbfgs_memory'] == 10 and cfg['quench_optimizer'] == 'safe-lbfgs-total'
        checks = []
        trace = []
        gates = Counter()
        event_status = Counter()
        if result is not None:
            assert result['evaluation_requests'] == len(paid)
            initial_cost = result['initial']['evaluation_requests']
            assert initial_cost + sum(x['evaluation_requests'] for x in result['records']) == len(paid)
            assert len(result['minima']) == len(fresh)
            for minimum, check in zip(result['minima'], fresh):
                good = ('error' not in check and check['force_qualified'] and check['cell_exact']
                        and check['pbc_exact'] and abs(check['energy_error']) <= 1e-8)
                if 'forces' in check:
                    assert np.isclose(np.linalg.norm(check['forces'], axis=1).max(), check['fmax'], rtol=0, atol=1e-12)
                checks.append(bool(good))
            cost = initial_cost
            trace.append((cost, result['initial']['energy']))
            for record in result['records']:
                cost += record['evaluation_requests']
                landing = record['landing']
                if landing is not None and landing['converged']:
                    trace.append((cost, landing['energy']))
                for event in record['climb']:
                    event_status[event.get('status', 'missing')] += 1
                    decision = event.get('diagnostics', {}).get('decision', {})
                    for key in ('force_stop', 'e_limit_stop', 'energy_lower', 'saved_energy_stop', 'step_over', 'final_ng_stop'):
                        gates[key] += int(decision.get(key, False))
            best_minimum = min(result['minima'], key=lambda x: x['energy'])
            assert result['best'] == best_minimum['atoms']
            best = best_minimum['energy']
            assert np.isclose(best, min(e for _, e in trace), rtol=0, atol=1e-12)
        else:
            best = None
        key = (summary['case'], summary['strategy'], summary['arm'])
        traces[key] = trace
        rows.append(dict(case=key[0], strategy=key[1], arm=key[2], status=summary['status'],
                         search_requests=len(paid), fresh_requests=summary['fresh']['requests'],
                         paid_failures=sum(x['kind'] == 'failure' for x in paid),
                         denials=[x.get('error') for x in ledger if not x['charged']],
                         minima=len(checks), qualified=sum(checks), best_energy=best,
                         record_statuses=[] if result is None else [x['status'] for x in result['records']],
                         accepted=0 if result is None else sum(x['accepted'] for x in result['records']),
                         event_statuses=dict(event_status), gates=dict(gates),
                         fragments=[x.get('connectivity') for x in fresh],
                         search_wall_seconds=summary['wall_seconds'], fresh_wall_seconds=summary['fresh']['wall_seconds']))
    assert len(rows) == 12, 'incomplete campaign'
    comparisons = []
    for case in ('cu13', 'cu31_fixed', 'bicyclobutane'):
        for strategy in ('forward_force', 'pam_height_width'):
            pair = [next(x for x in rows if (x['case'], x['strategy'], x['arm']) == (case, strategy, arm))
                    for arm in ('baseline', 'stage_control')]
            cap = min(x['search_requests'] for x in pair)
            energies = []
            for arm in ('baseline', 'stage_control'):
                values = [e for c, e in traces[(case, strategy, arm)] if c <= cap]
                energies.append(min(values) if values else None)
            comparisons.append(dict(case=case, strategy=strategy, common_requests=cap,
                                    baseline_energy=energies[0], stage_energy=energies[1],
                                    stage_minus_baseline=None if None in energies else energies[1]-energies[0]))
    report = dict(arms=rows, matched_prefix=comparisons,
                  total_search=sum(x['search_requests'] for x in rows),
                  total_fresh=sum(x['fresh_requests'] for x in rows),
                  all_stored_minima_qualified=all(x['minima'] > 0 and x['minima'] == x['qualified'] for x in rows),
                  note='Single-seed two-attempt development probe; no basin uniqueness or GM claim.')
    (root / 'root-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))
