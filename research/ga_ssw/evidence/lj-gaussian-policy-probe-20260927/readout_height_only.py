"""Join the prospective height-only follow-up to its saved paired controls."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(followup, controls):
    new=json.loads((followup/'summary.json').read_text())
    old=json.loads((controls/'summary.json').read_text())
    assert len(new['rows'])==4 and new['search_requests_total']<=10000 and new['fresh_requests_total']<=8
    index={(r['n'],r['seed'],r['policy']):r for r in old['rows']}
    rows=[]
    for row in new['rows']:
        n,seed=row['n'],row['seed'];base=index[n,seed,'forward_default']
        # Check effective settings, initial state and actual first mode, not just a common seed.
        assert row['ssw_config']==base['ssw_config']
        assert row['recovered_rotation']==base['recovered_rotation']
        assert row['search_rng_seed_sequence']==base['search_rng_seed_sequence']
        assert row['initial']['energy_eV']==base['initial']['energy_eV']
        a,b=row['first_gaussian'],base['first_gaussian']
        assert np.array_equal(a['center'],b['center']) and np.array_equal(a['direction'],b['direction'])
        assert row['gaussian_policy']['mode']=='height_only'
        folder=followup/f'lj{n}-seed{seed}'/row['policy']
        records=json.loads((folder/'records.json').read_text());rec=records[0]
        assert all(stage.get('width',.6)==.6 for stage in rec['climb'])
        fresh={item['label']:item for item in row['fresh_checks']}
        start,end=fresh['initial'],fresh['landing']
        assert start['status']==end['status']=='evaluated'
        rows.append(dict(n=n,seed=seed,status=rec['status'],accepted=rec['accepted'],
            search=row['search_requests'],fresh=row['fresh_requests'],boundary=row['boundary'],
            force_qualified=start['fmax_eV_A']<=.01 and end['fmax_eV_A']<=.01,
            delta_energy_eV=end['energy_eV']-start['energy_eV'],
            connected=all(v['connected'] for v in end['connectivity']['thresholds'].values()),
            first_height=a['weight'],first_width=a['width'],stages=len(rec['climb']),
            clipping=row['clipping_counts'],paired_with_controls=True))
    assert sum(r['search'] for r in rows)==new['search_requests_total']
    assert sum(r['fresh'] for r in rows)==new['fresh_requests_total']
    return dict(scope='Development follow-up; no long-search or pure-height universal claim',
                followup=str(followup.resolve()),controls=str(controls.resolve()),
                search=new['search_requests_total'],fresh=new['fresh_requests_total'],rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=compare(a.followup,a.controls)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
