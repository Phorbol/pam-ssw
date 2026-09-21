"""Offline full-denominator report for the frozen TYPE4 held-out pilot."""
import json
from pathlib import Path
from research.ga_ssw.audit_multistep_state import audit_payload


def main():
    root=Path(__file__).resolve().parent/'evidence/type4-multistep-heldout'
    rows=[]
    for seed in (7,101):
        for arm in ('physical_mask_only','source_direction_mask'):
            path=root/f'seed{seed}'/arm/'result.json'
            if not path.exists():
                rows.append(dict(seed=seed,arm=arm,status='not_finalized'));continue
            d=json.loads(path.read_text());r=d.get('result');row=dict(seed=seed,arm=arm,total_requests=d['total_requests'],search_requests=d.get('search_requests'),seconds=d['seconds'],source=str(path.relative_to(root)),error=d.get('error'))
            if r is None:row['status']='failed_no_result';rows.append(row);continue
            audit=audit_payload(d);(path.parent/'state-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            initial=r['initial']['energy'];events=r['records'][1:];cumulative=r['records'][0]['requests'];best=initial;envelope=[]
            for event in events:
                cumulative+=event['requests'];landing=event.get('landing');valid=event['status']=='valid_landing'
                if valid:best=min(best,landing['energy'])
                envelope.append(dict(index=event['index'],status=event['status'],accepted=event['accepted'],cumulative_search_requests=cumulative,best_energy=best,best_delta=best-initial,landing_energy=landing.get('energy') if landing else None,biased_stages=len(event['climb']),stage_requests=[s['requests'] for s in event['climb']]))
            row.update(status=r['status'],audit_status=audit['status'],attempts=len(events),valid_landings=sum(e['status']=='valid_landing' for e in events),accepted=sum(e['accepted'] for e in events),initial_energy=initial,best_energy=best,best_delta=best-initial,envelope=envelope,fresh_count=len(d.get('fresh',[])),fresh_max_fmax=max((f['active_fmax'] for f in d.get('fresh',[])),default=None),fresh_max_energy_error=max((abs(f['energy_error']) for f in d.get('fresh',[])),default=None),all_fresh_invariants=all(f['fixed_exact'] and f['cell_exact'] for f in d.get('fresh',[])))
            rows.append(row)
    output=dict(planned_arms=4,planned_attempts=12,finalized_arms=sum(r['status']!='not_finalized' for r in rows),rows=rows,total_recorded_search_and_fresh_requests=sum(r.get('total_requests',0) for r in rows),scope='two held-out seeds, three requested attempts per arm and equal3000EF cap; envelope records actual landed candidates only; fresh force checks are not Hessian or physical validity; no basin-identity threshold or efficiency rank inferred')
    (root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
