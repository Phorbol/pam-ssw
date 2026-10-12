"""Read-only algebra check of native generator snapshots; no PES operations."""
import argparse
import json
import math
from pathlib import Path


def flat(snapshot, key):
    descriptor = snapshot[key]
    if descriptor['status'] != 'read':
        raise ValueError('vector not recorded')
    return [v for row in descriptor['values'] for v in row]


def compare(actual, expected):
    if len(actual) != len(expected):
        raise ValueError('vector size mismatch')
    return dict(max_abs_error=max(abs(a-b) for a,b in zip(actual, expected)),
                expected_norm=math.hypot(*expected))


def analyze(run):
    process = json.loads((run/'process.json').read_text())
    if process['state'] != 'completed' or process['returncode'] != 0 or process['cleanup_survivors']:
        raise ValueError('native trace did not complete cleanly')
    if (run/'trace-error.json').exists() or (run/'trace-stop.json').exists():
        raise ValueError('trace error or cap prevents complete comparison')
    stages = {}
    for line in (run/'trace.jsonl').read_text().splitlines():
        row=json.loads(line)
        snap=row.get('generator_snapshot')
        if snap:
            phases=stages.setdefault(snap['generator_serial'], {})
            if snap['phase'] in phases:
                raise ValueError('duplicate snapshot phase')
            phases[snap['phase']]=snap
    result=[]
    dest='destination_object_offset1788'
    scratch='scratch_object_offset1848'
    for serial, phases in stages.items():
        entry=phases['entry']; coeff=entry['mode_coefficients']
        if any(coeff[i] != 0 for i in (0,1,2,4,6,7,8)):
            raise ValueError('additional active coefficient invalidates three-term check')
        values={phase:flat(snap,dest) for phase,snap in phases.items()}
        record=dict(generator_serial=serial, coefficients=coeff,lrigid=entry['lrigid_raw'],
                    phases=list(phases),entry_norm=(math.hypot(*values['entry']) if coeff[9]>1e-6 else None),
                    entry_interpretation=('active_history' if coeff[9]>1e-6 else
                                          'history_disabled_entry_storage_not_interpreted'))
        history=values['after_history_scale']
        # Native only scales when coefficient is positive; zero skips this block.
        record['history']=compare(history,[coeff[9]*x for x in values['entry']]) if coeff[9]>1e-6 else None
        for phase, previous, index in [('after_coefficient18','after_history_scale',3),
                                       ('after_coefficient28','after_coefficient18',5)]:
            contribution=flat(phases[phase],scratch)
            if len(contribution) != len(values[previous]):
                raise ValueError('scratch size mismatch')
            record[phase]=compare(values[phase],[a+coeff[index]*b for a,b in zip(values[previous],contribution)])
        result.append(record)
    return dict(run=str(run),process=process,stages=result,
                scope='native vector algebra only; not search efficacy or block-cycle equivalence')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args(); result=analyze(args.run)
    args.out.mkdir(exist_ok=False)
    (args.out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
