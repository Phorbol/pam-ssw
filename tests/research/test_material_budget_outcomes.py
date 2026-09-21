"""All eight actual gate artifacts retain all sixteen planned outer slots."""
import collections,json
from pathlib import Path
from research.ga_ssw.material_budget_outcomes import classify_material_events


def test_actual_eight_run_ledger_outcomes_and_paid_intervals():
    root=Path(__file__).resolve().parents[2]/'research/ga_ssw/prospective/complex-vc-feasibility/results'
    files=list(root.glob('*/result.json'));assert len(files)==8
    total=0;counts=collections.Counter()
    for p in files:
        d=json.loads(p.read_text());rows=classify_material_events(d)
        assert len(rows)==d['steps_requested']==2
        total+=d['requests'];counts.update(x['outcome'] for x in rows.values())
        assert sum(e['requests'] for e in d['records'])==d['requests']
        assert rows[0]['request_interval'][0]==d['initial_requests']
        assert rows[1]['request_interval'][1]==d['requests']
        if p.parent.name=='brookite48-joint-seed17':
            assert rows[0]['outcome']=='budget_censored'
            assert rows[0]['request_interval']==[10,2000]
            assert rows[1]['outcome']=='not_started_no_budget'
            assert rows[1]['request_interval']==[2000,2000]
        for x in rows.values():
            if x['outcome']=='budget_censored':assert x['budget_ledger_evidence']
    assert total==15467
    assert counts==dict(valid_accepted=1,valid_rejected=7,budget_censored=7,not_started_no_budget=1)
