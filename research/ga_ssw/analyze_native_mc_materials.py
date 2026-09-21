"""Independently reconstruct recorded MC decisions and reconcile material costs."""
import argparse
import json
import math
from pathlib import Path


def analyze(root):
    plan = json.loads((root / 'plan.json').read_text())
    rows = []
    for case in plan['cases']:
        folder = root / case
        result = json.loads((folder / 'result.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        qualification = json.loads((folder / 'fresh-qualification.json').read_text())
        current = result['initial']['energy']
        nsame = 0
        issues = []
        decisions = 0
        uphill = 0
        for record in result['records']:
            mc = record['mc_telemetry']
            if mc is None:
                if record['landing'] is not None and record['landing']['converged']:
                    issues.append(f"record {record['index']}: qualified landing without decision")
                continue
            decisions += 1
            delta = record['landing']['energy'] - current
            near = abs(delta) < plan['native_mc']['energy_tol_eV']
            count = nsame + int(near)
            exponent = count - plan['native_mc']['maxtrap']
            if exponent >= 0:
                raise ValueError('This experiment excludes heating; use a separately qualified analyzer for it')
            temperature = plan['configs'][case]['temperature_K']
            probability = 1. if delta <= 0 else math.exp(-delta * 96485. / (20. * 8.314 * temperature))
            accept = delta <= 0 or mc['uniform'] <= probability
            expected_state = 0 if accept and not near else count
            checks = [math.isclose(mc['delta_energy_eV'], delta, abs_tol=1e-12),
                      math.isclose(mc['acceptance_probability'], probability, rel_tol=1e-12, abs_tol=1e-15),
                      mc['effective_temperature_K'] == temperature,
                      mc['temperature_increment_K'] == 0,
                      mc['nsame_for_acceptance'] == count,
                      mc['near_equal_energy'] == near,
                      mc['state']['nsame'] == expected_state,
                      mc['accepted'] == accept == record['accepted']]
            if not all(checks):
                issues.append(f"record {record['index']}: decision mismatch {checks}")
            uphill += int(delta > 0)
            nsame = expected_state
            if accept:
                current = record['landing']['energy']
        ledger_count = sum(1 for line in (folder / 'requests.jsonl').open() if line.strip())
        if ledger_count != summary['search_requests'] or ledger_count != result['evaluation_requests']:
            issues.append('request accounting mismatch')
        if decisions != summary['mc_decisions']:
            issues.append('decision count mismatch')
        if summary['fresh_requests'] != len(qualification):
            issues.append('fresh request accounting mismatch')
        rows.append(dict(case=case, status=result['status'], decisions=decisions, uphill_decisions=uphill,
                         search_requests=ledger_count, fresh_requests=summary['fresh_requests'],
                         fresh_qualified=sum(bool(q['qualified']) for q in qualification),
                         best_energy_eV=min(m['energy'] for m in result['minima']), issues=issues))
    return dict(scope='MC integration qualification, not search efficacy', rows=rows,
                total_search_requests=sum(r['search_requests'] for r in rows),
                total_fresh_requests=sum(r['fresh_requests'] for r in rows),
                all_decisions_and_costs_match=all(not r['issues'] for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    with (args.directory / 'analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))
