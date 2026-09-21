"""Offline campaign audit: paid API ledger, fresh certificates, matched prefixes."""
import argparse,json
from pathlib import Path
import numpy as np
from ase import Atoms
from research.ga_ssw.audit_aloh_optimizer_geometry import describe

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);root=ap.parse_args().root
    rows=[]
    for p in sorted(root.glob('frame*/*/result.json')):
        r=json.loads(p.read_text()); s=json.loads((p.parent/'summary.json').read_text())
        fresh=json.loads((p.parent/'fresh.json').read_text()); count=0;denials=0
        with (p.parent/'ledger.jsonl').open() as f:
            for line in f:
                e=json.loads(line)
                if e.get('charged'):
                    count+=1;assert e['after']==count
                denials+=int(e['kind']=='denial')
        assert count==s['search_requests']==r['evaluation_requests']
        initial=r['initial']; cumulative=initial['evaluation_requests']
        timeline=[dict(request=cumulative,index=0,energy=initial['energy'])]; idx=1
        for record in r['records']:
            cumulative+=record['evaluation_requests']
            if record['landing'] and record['landing']['converged']:
                m=r['minima'][idx]
                assert abs(m['energy']-record['landing']['energy'])<1e-10
                timeline.append(dict(request=cumulative,index=idx,energy=m['energy']));idx+=1
        assert cumulative==count and idx==len(r['minima'])
        checks=[];geometries=[]
        assert len(fresh)==len(r['minima'])
        for i,(m,f) in enumerate(zip(r['minima'],fresh)):
            assert f['index']==i
            a,b=m['atoms'],initial['atoms']
            invariant=a['numbers']==b['numbers'] and a['pbc']==b['pbc'] and np.array_equal(a['cell'],b['cell'])
            qualified=invariant and f['status']=='checked' and np.isfinite(f['energy_error']) and abs(f['energy_error'])<=1e-7 and np.isfinite(f['fmax']) and f['fmax']<=.03
            checks.append(bool(qualified))
            atoms=Atoms(numbers=a['numbers'],positions=a['positions'],cell=a['cell'],pbc=a['pbc'])
            geometries.append(dict(index=i,energy=m['energy'],qualified=bool(qualified),**describe(atoms)))
        best=min(range(len(geometries)),key=lambda i:geometries[i]['energy'])
        hsets={tuple((x['h'],x['o']) for x in g['hydrogen_oxygen']) for g in geometries}
        rows.append(dict(arm=p.parent.name,frame=s['frame'],optimizer=s['optimizer'],status=s['status'],
            search_requests=count,fresh_requests=len(fresh),qualified=sum(checks),observations=len(checks),
            completed_outer=len(r['records'])-denials,record_count=len(r['records']),accepted=sum(x['accepted'] for x in r['records']),
            seconds=s['elapsed_seconds'],initial_energy=initial['energy'],best_energy=geometries[best]['energy'],
            best_delta=geometries[best]['energy']-initial['energy'],timeline=timeline,
            labeled_HO_patterns=len(hsets),geometry=geometries,best_geometry=geometries[best]))
    comparisons=[]
    for frame in (0,1):
        group=[r for r in rows if r['frame']==frame];prefix=min(r['search_requests'] for r in group)
        for r in group:
            t=[x for x in r['timeline'] if x['request']<=prefix];best=min(t,key=lambda x:x['energy'])
            comparisons.append(dict(frame=frame,optimizer=r['optimizer'],common_search_prefix=prefix,
                observations_at_prefix=len(t),best_energy=best['energy'],best_delta=best['energy']-r['initial_energy']))
    out=root/'root-completed-audit.json';assert not out.exists()
    out.write_text(json.dumps(dict(rows=rows,comparisons=comparisons,scope='Stored observations are not unique basins; labeled HO patterns are not permutation invariant. All geometry analysis is offline; no new PES requests.'),indent=2)+'\n')
    print(json.dumps(dict(total_search=sum(r['search_requests'] for r in rows),fresh=sum(r['fresh_requests'] for r in rows),qualified=sum(r['qualified'] for r in rows),comparisons=comparisons)))
if __name__=='__main__':main()
