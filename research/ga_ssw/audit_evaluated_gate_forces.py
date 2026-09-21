"""Recompute recorded gate snapshots from paid physical E/F and bias history."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pamssw.standalone.gaussian import ProjectedGaussian


def audit(root):
    runs=[]
    for path in sorted(root.glob('*/*/result.json')):
        result=json.loads(path.read_text())
        ledger={}
        for line in (path.parent/'requests.jsonl').open():
            row=json.loads(line)
            if row['kind']=='search': ledger[row['request']]=row
        before=result['initial']['evaluation_requests']; checks=[]
        for record in result['records']:
            end=before; terms=[]
            for event in record['climb']:
                if 'requests' not in event:
                    assert event.get('stage_stop_reason')!='adapter'
                    checks.append(dict(checked=False,reason='incomplete terminal event has no stage request delta'))
                    break
                end+=event['requests']
                if event.get('stage_stop_reason')!='adapter': continue
                terms.append(ProjectedGaussian(np.array(event['center']),np.array(event['direction']),
                                               event['width'],event['weight']))
                if 'true_energy' not in event:
                    checks.append(dict(checked=False,reason='no post-stage true evaluation'));continue
                row=ledger[end-1]
                atoms=SimpleNamespace(positions=np.array(row['atoms']['positions']))
                energy=row['energy']; forces=np.array(row['forces'])
                for term in terms:
                    de,df=term.evaluate(atoms);energy+=de;forces+=df
                norm=float(np.linalg.norm(forces,axis=1).max());component=float(np.abs(forces).max())
                diagnostic=event['diagnostics'];decision=diagnostic['decision']
                errors=[abs(energy-event['biased_energy']),abs(norm-event['max_force']),
                        abs(component-decision['max_force'])]
                assert max(errors)<1e-8,(path,record['index'],event['index'],errors)
                assert bool(component<decision['climb_stopf'])==decision['force_stop']
                checks.append(dict(checked=True,record=record['index'],gaussian=event['index'],
                                   request=row['request'],errors=errors,decision=decision))
            before+=record['evaluation_requests']
        runs.append(dict(run=path.parent.name,checks=checks))
    return dict(scope='Independent algebraic reconstruction from recorded physical force, not a new oracle evaluation.',runs=runs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=audit(a.input)
    with a.output.open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')
    print([(r['run'],sum(c['checked'] for c in r['checks'])) for r in data['runs']])
