"""Offline audit of the four frozen native VC optimizer cases; no PES calls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def read(path): return json.loads(Path(path).read_text())

def audit(root, reference):
    rows=[]
    for arm in ('ls_all','ls_filter'):
        for seed in (7,101):
            path=root/'diagnostic'/f'{arm}-seed{seed}.json'
            native=read(path)
            assert native['source_sha256']==hashlib.sha256(Path(native['source_path']).read_bytes()).hexdigest()
            assert native['total_requests']==native['optimizer_requests']+native['fresh_requests']<=330
            assert native['fresh_requests']==1
            assert native['optimizer_requests']==len(native['evaluations'])
            assert native['accepted_steps']==len(native['accepted'])<=300
            assert max(native['initial_agreement'].values())<=1e-8
            assert all(a['accepted_coordinate_error']<=1e-12 for a in native['accepted'])
            final=native['final_fresh']; assert final['status']=='checked'
            accepted=native['accepted'][-1] if native['accepted'] else native['evaluations'][0]
            assert np.array_equal(final['source_q'],accepted['q'])
            energy_error=abs(final['objective']-accepted['objective'])
            gradient_error=float(np.max(np.abs(np.array(final['gradient'])-accepted['gradient'])))
            assert max(energy_error,gradient_error)<1e-8
            safe=[]
            for h in (10,500):
                oldpath=reference/f'seed{seed}'/arm/f'history{h}'/'result.json'
                old=read(oldpath); initial=old['safe_lbfgs']['trace'][0]
                assert np.array_equal(native['q_start'],old['q_start'])
                assert abs(native['evaluations'][0]['objective']-initial['energy'])<1e-8
                assert np.max(np.abs(np.array(native['evaluations'][0]['gradient'])-initial['gradient']))<1e-8
                f=old['final_fresh']; assert f['status']=='checked'
                safe.append(dict(history=h,source=str(oldpath),status=old['safe_lbfgs']['status'],
                    requests=old['requests'],steps=old['safe_lbfgs']['steps'],norm=f['norm'],
                    qualified=f['norm']<=.001))
            rows.append(dict(arm=arm,seed=seed,source=str(path),status=native['status'],
                requests=native['total_requests'],optimizer_requests=native['optimizer_requests'],
                fresh_requests=native['fresh_requests'],accepted_steps=native['accepted_steps'],
                norm=final['vc_norm'],qualified=final['vc_norm']<=.001,
                initial_agreement=native['initial_agreement'],fresh_energy_error=energy_error,
                fresh_gradient_error=gradient_error,safe=safe))
    return dict(cases=rows,requested_cases=4,total_requests=sum(r['requests'] for r in rows),
        qualified=sum(r['qualified'] for r in rows),
        comparison='same frozen objectives/starts/gradient gate/budget; different native and Safe-total step/history configurations',
        boundary='biased stationary-point qualification only, not true minima or SSW efficacy; no new PES during this audit')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    result=audit(a.root,a.reference)
    (a.root/'diagnostic/audit-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('requested_cases','total_requests','qualified')}))
    for r in result['cases']: print(r['arm'],r['seed'],r['status'],r['requests'],r['norm'])
