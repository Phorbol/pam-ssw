"""Read-only ledger and tolerance-certificate summary for the Fe7C3 grid."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def read_rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def _fmax_config(d, plan):
    """Extract declared tolerances; missing fields remain None."""
    ac = d.get('atomic_config') or {}
    bc = d.get('block_config') or {}
    jc = d.get('joint_config') or {}
    ba = bc.get('atomic') if isinstance(bc.get('atomic'), dict) else {}
    pa = plan.get('atomic') or {}
    pj = plan.get('joint') or {}
    return {
        'inner_fmax': ac.get('bias_fmax', pa.get('bias_fmax')),
        'outer_fmax': jc.get('fmax', pj.get('fmax', ba.get('fmax', pa.get('fmax')))),
        'partial_atom_fmax': bc.get('partial_atom_fmax', (plan.get('block') or {}).get('partial_atom_fmax')),
    }

def _failure_entries(obj, path=''):
    """Return nonempty failure/error explanations with their JSON paths."""
    out = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f'{path}/{key}'
            if key in {'error', 'message', 'reason'} and value not in (None, '', [], {}):
                text = str(value)
                low = text.lower()
                if any(t in low for t in ('fail', 'error', 'budget', 'maxiter', 'request', 'exhaust', 'denied')):
                    out.append({'path': here, 'value': value})
            if key != 'kernel':  # kernel mirrors records and would double the evidence
                out.extend(_failure_entries(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.extend(_failure_entries(value, f'{path}/{i}'))
    return out

def arm(path, plan):
    rp,ep=path/'result.json',path/'evaluations.jsonl'
    if not rp.is_file() or not ep.is_file(): return {'status':'pending','missing':[x.name for x in (rp,ep) if not x.is_file()]}
    d=json.loads(rp.read_text()); rows=read_rows(ep)
    charged=sum(bool(x.get('charged')) for x in rows); search=sum(x.get('stage')=='search' and bool(x.get('charged')) for x in rows); fresh=sum(x.get('stage')=='fresh' and bool(x.get('charged')) for x in rows); denied=len(rows)-charged
    errors=[]
    if d.get('requests') is not None and int(d['requests'])!=charged: errors.append('reported_requests_vs_charged')
    records=d.get('records') or []; statuses=[]
    for r in records:
        if isinstance(r,dict):
            statuses.append(r.get('status','initial'))
            for c in (r.get('climb') or []):
                if isinstance(c,dict): statuses.append(c.get('status'))
    completed=[]
    for r in records:
        atomic=(r.get('atomic') or {}) if isinstance(r,dict) else {}
        cp=atomic.get('checkpoint') or {}
        events=atomic.get('climb') or cp.get('climb') or r.get('climb') or []
        completed.extend(x for x in events if isinstance(x,dict) and 'true_energy' in x)
    land=d.get('landings') or []; initial=land[0] if land else None
    landings=[]
    for x in land:
        if int(x.get('index',-1))>=0:
            e, e0 = x.get('energy'), initial.get('energy') if initial else None
            delta = e - e0 if isinstance(e, (int, float)) and isinstance(e0, (int, float)) else None
            landings.append({'index':x.get('index'),'accepted':bool(x.get('accepted')),'energy':e,'energy_delta_from_initial':delta})
    checks=(d.get('fresh') or {}).get('checks') or []; fresh_rows=[]
    for c in checks:
        ci = c.get('index')
        li = land[int(ci)].get('index') if isinstance(ci, int) and 0 <= ci < len(land) else None
        fresh_rows.append({'check_index':ci,'landing_index':li,'certified':c.get('certified') is True,'fmax':c.get('fmax'),'stress_max':c.get('stress_max'),'energy':c.get('energy'),'force_error':c.get('force_error'),'stress_error':c.get('stress_error')})
    cross={}
    for ft in (.001,.01,.05):
        cross[str(ft)]=[{'check_index':c.get('index'),'landing_index':(land[int(c['index'])].get('index') if isinstance(c.get('index'), int) and 0 <= c['index'] < len(land) else None),'certified':(isinstance(c.get('fmax'), (int, float)) and isinstance(c.get('stress_max'), (int, float)) and c['fmax']<=ft and c['stress_max']<=.0001)} for c in checks]
    budget=[s for s in statuses if s and ('budget' in str(s).lower() or 'maxiter' in str(s).lower() or 'request' in str(s).lower())]
    failure_reasons = _failure_entries(d)
    for i, row in enumerate(rows):
        failure_reasons.extend(_failure_entries(row, f'/evaluations.jsonl/{i}'))
    # Preserve order while avoiding repeated identical evidence from nested mirrors.
    seen = set(); failure_reasons = [x for x in failure_reasons if not ((x['path'], str(x['value'])) in seen or seen.add((x['path'], str(x['value']))))]
    budget_evidence = [{'path': f'/records/status/{i}', 'value': s} for i, s in enumerate(budget)]
    budget_evidence.extend(x for x in failure_reasons if any(t in str(x['value']).lower() for t in ('budget', 'maxiter', 'request', 'exhaust', 'denied')))
    reported_search = d.get('search_requests')
    if isinstance(reported_search, int) and reported_search != search: errors.append('reported_search_requests_vs_search_charged')
    reported_fresh = (d.get('fresh') or {}).get('charged')
    if isinstance(reported_fresh, int) and reported_fresh != fresh: errors.append('reported_fresh_charged_vs_ledger')
    return {'status':'audited','seed':d.get('seed'),'config':_fmax_config(d, plan),'counts':{'rows':len(rows),'charged':charged,'search_charged':search,'fresh_charged':fresh,'denied':denied,'search_denied':sum(x.get('stage')=='search' and not x.get('charged') for x in rows),'fresh_denied':sum(x.get('stage')=='fresh' and not x.get('charged') for x in rows)},'reported_requests':d.get('requests'),'reported_search_requests':reported_search,'consistency_errors':errors,'record_count':len(records),'record_statuses':statuses,'completed_gaussian_count':len(completed),'landings':landings,'fresh':{'requested':(d.get('fresh') or {}).get('requested'),'checked':len(checks),'certified':sum(c.get('certified') is True for c in checks),'checks':fresh_rows},'cross_certificates_fmax_stress':cross,'budget_failure_reasons':budget_evidence,'failure_reasons':failure_reasons,'reported_status':d.get('status')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--output',type=Path);a=ap.parse_args(); root=a.root; arms={}
    for metric in sorted(root.iterdir()):
        planp=metric/'plan.json'
        if not planp.is_file(): continue
        plan=json.loads(planp.read_text()); solver=(plan.get('research_solvers') or ['safe_total'])[0]
        for seed in plan.get('research_seeds',plan.get('seeds',[7,101])):
            path = metric/'comparison'/f'{solver}-seed{seed}'
            used = path/'plan-used.json'
            arms[f'{metric.name}/{solver}-seed{seed}']=arm(path, json.loads(used.read_text()) if used.is_file() else plan)
    out={'status':'audited' if arms and all(x['status']=='audited' for x in arms.values()) else 'pending','zero_pes':True,'expected_arms':len(arms),'arms':arms,'certificate_definition':{'fmax_thresholds':[.001,.01,.05],'stress_threshold':.0001},'scope':'ledger and force/stress certificate summary; no structural identity or new PES'}
    p=a.output or root/'tolerance-summary.json';p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(p)
if __name__=='__main__':main()
