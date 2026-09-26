"""Secondary readout of the frozen four-arm study; no new PES or search."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(source):
    audit = json.loads(source.read_text())
    assert not audit['errors'] and len(audit['arms']) == 4
    rows=[]
    for arm in audit['arms']:
        folder=Path(arm['folder'])
        result=json.loads((folder/'result.json').read_text())
        qualified=iter(arm['result']['minima'][1:])
        counts=Counter(); statuses=Counter(); costs=Counter()
        costs['initial']=arm['result']['minima'][0]['cost_request']
        current_fragmented=False
        assert arm['result']['minima'][0]['graph']['1.8']['components']==1
        accepted_fragment_transitions=0
        for record in result['records']:
            statuses[record['status']]+=1
            landing=record['landing']
            if landing is None or not landing['converged']:
                counts['without_qualified_landing']+=1
                costs['without_qualified_landing']+=record['evaluation_requests']
                continue
            metric=next(qualified)
            assert metric['energy_eV']==landing['energy']
            counts['qualified_landings']+=1
            flags={key: metric['graph'][key]['components']>1 for key in ('1.64','1.7','1.8')}
            assert len(set(flags.values()))==1, 'Threshold-sensitive; revise reporting before combining'
            fragmented=flags['1.8']
            counts['fragmented_landings' if fragmented else 'connected_landings']+=1
            costs['fragmented_landings' if fragmented else 'connected_landings']+=record['evaluation_requests']
            if record['accepted']:
                counts['accepted_landings']+=1
                if fragmented:
                    counts['accepted_fragmented']+=1
                    if not current_fragmented:accepted_fragment_transitions+=1
                current_fragmented=fragmented
            if current_fragmented:counts['outer_steps_current_fragmented']+=1
        assert next(qualified,None) is None
        assert sum(costs.values())==arm['search_requests']
        curve=arm['result']['best_energy_curve']
        half=[x for x in curve if x['cost_request']<=30000]
        assert half
        rows.append(dict(case=arm['case'],method=folder.parent.name,source=str(folder),
            boundary=arm['boundary'],requests=arm['search_requests'],counts=dict(counts),statuses=dict(statuses),costs=dict(costs),
            accepted_fragment_transitions=accepted_fragment_transitions,
            best_improvement_second_half_eV=half[-1]['best_energy_eV']-curve[-1]['best_energy_eV'],
            any_target_cage=arm['result']['any_target_cage']))
    return dict(scope='Post-hoc failure-layer audit of existing held-out study; no new independent trials',
                source=str(source.resolve()),extra_pes_calls=0,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=analyze(Path(__file__).with_name('analysis-1440586.json'))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
