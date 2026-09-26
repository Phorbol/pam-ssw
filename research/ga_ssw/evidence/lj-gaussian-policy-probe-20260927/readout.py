"""Zero-PES readout; pairing and baseline replay precede policy interpretation."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read


def same_label_rms(a, b):
    x = a.positions - a.positions.mean(axis=0)
    y = b.positions - b.positions.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum((x @ (u @ sign @ vt) - y)**2, axis=1))))


def readout(root, previous):
    data = json.loads((root/'summary.json').read_text())
    assert len(data['rows']) == 8
    assert data['search_requests_total'] <= 20000
    assert data['fresh_requests_total'] <= 16
    rows, failures, replay = [], [], []
    for row in data['rows']:
        path = root/f"lj{row['n']}-seed{row['seed']}"/row['policy']
        records = json.loads((path/'records.json').read_text()) if (path/'records.json').exists() else []
        rec = records[0] if records else {}
        fresh = {item['label']:item for item in row.get('fresh_checks',[])}
        start, end = fresh.get('initial',{}), fresh.get('landing',{})
        valid = all(item.get('status')=='evaluated' and item['fmax_eV_A']<=.01
                    for item in (start,end))
        connected = bool(end.get('connectivity')) and all(
            value['connected'] for value in end['connectivity']['thresholds'].values())
        first = row.get('first_gaussian') or {}
        heights = [s['weight'] for s in rec.get('climb',[]) if 'weight' in s]
        widths = [s['width'] for s in rec.get('climb',[]) if 'width' in s]
        delta = end.get('energy_eV',0)-start.get('energy_eV',0) if valid else None
        rms = same_label_rms(read(path/'initial.extxyz'),read(path/'landing.extxyz')) if valid else None
        rows.append(dict(n=row['n'],seed=row['seed'],policy=row['policy'],status=row['status'],
            outer_status=rec.get('status'),accepted=rec.get('accepted'),boundary=row.get('boundary'),
            search=row['search_requests'],fresh=row['fresh_requests'],force_qualified=valid,
            connected=connected,delta_energy_eV=delta,same_label_proper_rms_A=rms,
            rms_interpretation='Small RMS supports same geometry; large RMS does not exclude a permutation',
            first_height=first.get('weight'),first_width=first.get('width'),
            max_height=max(heights) if heights else None,min_width=min(widths) if widths else None,
            stages=len(rec.get('climb',[])),clipping=row.get('clipping_counts')))
        if row['n']==38 and row['policy']=='forward_default' and records:
            old=json.loads((previous/f"lj38-paper-seed{row['seed']}"/'summary.json').read_text())
            out=old['outer_steps'][0]
            checks=dict(status=rec['status']==out['status'],accepted=rec['accepted']==out['accepted'],
                cost=row['search_requests']==old['initial_requests']+out['outer_requests'],
                energy=abs(row['landing']['energy_eV']-out['landing_energy_eV'])<1e-9)
            replay.append(dict(seed=row['seed'],checks=checks))
            if not all(checks.values()):failures.append(dict(kind='old_baseline_replay',seed=row['seed'],checks=checks))
    index={(r['n'],r['seed'],r['policy']):r for r in data['rows']}
    for check in data['pair_checks']:
        n,seed=check['n'],check['seed']
        a=index[n,seed,'forward_default']; b=index[n,seed,'pam_curvature_height_width']
        initial_match=(a.get('initial') is not None and b.get('initial') is not None and
                       abs(a['initial']['energy_eV']-b['initial']['energy_eV'])<1e-9)
        if not check['first_center_direction_match'] or not initial_match:
            failures.append(dict(kind='pairing',n=n,seed=seed,mode=check['first_center_direction_match'],initial=initial_match))
    assert sum(r['search'] for r in rows)==data['search_requests_total']
    assert sum(r['fresh'] for r in rows)==data['fresh_requests_total']
    return dict(source=str(root.resolve()),scope='Single-escape developer probe; no global-search ranking',
                search=data['search_requests_total'],fresh=data['fresh_requests_total'],
                pairing_or_replay_errors=failures,replay=replay,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=readout(a.run,a.previous)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
