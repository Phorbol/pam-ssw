"""Offline accounting and one-variable audit; no new calculator requests."""
import json
from collections import Counter
from pathlib import Path
from research.ga_ssw.summarize_fe7c3_joint_prequench import _primary


def read(path):
    return json.loads(path.read_text())


def arm(path):
    d = read(path/'result.json')
    ledger = Counter()
    for line in (path/'evaluations.jsonl').open():
        r = json.loads(line)
        if r.get('charged'):
            ledger[r['stage']] += 1
    assert sum(ledger.values()) == d['requests'] <= 2000
    assert ledger['search'] == d['search_requests'] == sum(r['requests'] for r in d['records'])
    assert len(d['records']) == 3
    assert d['fresh']['omitted'] == 0
    assert len(d['landings']) == len(d['fresh']['checks'])
    assert all(c['status'] == 'checked' and c['certified'] for c in d['fresh']['checks'])
    assert d['valid_proposals'] == len(d['landings'])-1
    return d, dict(requests=d['requests'], search_requests=d['search_requests'],
                   fresh_requests=ledger['fresh'], valid_proposals=d['valid_proposals'],
                   outcomes=dict(Counter(_primary(r) for r in d['records'][1:])),
                   attempt_costs=[r['requests'] for r in d['records'][1:]],
                   fresh_certified=len(d['fresh']['checks']),
                   climb_stages=[len(r.get('climb',[])) for r in d['records'][1:]])


def main():
    root=Path('research/ga_ssw/evidence/fe7c3-safe-total-inner-tolerance')
    old=Path(read(root/'plan.json')['source_control'])
    rows=[]
    for name in ['ls_all-seed7','ls_all-seed101','ls_filter-seed7','ls_filter-seed101']:
        a,ra=arm(old/'comparison'/name)
        b,rb=arm(root/'comparison'/name)
        diff={k:[a['joint_config'][k],v] for k,v in b['joint_config'].items() if v!=a['joint_config'][k]}
        assert diff=={'gradient_tol':[.001,.005]},diff
        assert a['ls']==b['ls']
        assert a['runtime']['model_sha256']==b['runtime']['model_sha256']
        rows.append(dict(case=name,strict=ra,existing_default=rb,config_difference=diff))
    out=dict(status='audited',new_PES=0,requested=8,rows=rows)
    for label in ['strict','existing_default']:
        out[label]=dict(requests=sum(r[label]['requests'] for r in rows),
            valid_proposals=sum(r[label]['valid_proposals'] for r in rows),
            outcomes=dict(sum((Counter(r[label]['outcomes']) for r in rows),Counter())))
    (root/'comparison'/'audit-summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
