"""Post-hoc two-scalar native guard replay; no PES and no species guard."""
import json, sys
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

run, out = map(Path, sys.argv[1:])
rows=[]
def closest(a):
    if a.pbc.any():
        # Expand only if no image pair is in range; includes nonzero self images.
        radius=3.0
        while True:
            d=neighbor_list('d',a,radius,self_interaction=False)
            if len(d): return float(np.min(d))
            radius *= 2
            if radius > 100: raise ValueError('no periodic neighbors')
    d=a.get_all_distances()
    np.fill_diagonal(d,np.inf)
    return float(d.min())

for folder in sorted(run.iterdir()):
    if not (folder/'result.json').exists(): continue
    result=json.loads((folder/'result.json').read_text())
    template=read(folder/'source-input.extxyz')
    stages=[]
    for record in result['records']:
        for stage in record['climb']:
            if 'width' not in stage: continue
            assert stage['cluster_frame'] in ('direction_only','translation_only')
            center=np.asarray(stage['center']); direction=np.asarray(stage['direction'])
            template.positions=center
            before=closest(template)
            trial=template.copy(); trial.positions=center+stage['width']*direction
            after=closest(trial)
            maximum=float(np.linalg.norm(trial.positions-center,axis=1).max())
            stages.append(dict(outer=record['index'],gaussian=stage['index'],maximum_atom_move_A=maximum,closest_before_A=before,closest_after_A=after,closest_ratio=after/before,rejected=bool(maximum>2.0 or after<0.75*before)))
    components=[]
    if not template.pbc.any():
        for index,a in enumerate(read(folder/'landings.extxyz',':')):
            dist=a.get_all_distances()
            for cutoff in (1.64,1.7,1.8):
                remaining=set(range(len(a))); groups=[]
                while remaining:
                    stack=[remaining.pop()]; group=[]
                    while stack:
                        i=stack.pop(); group.append(i)
                        new={j for j in remaining if dist[i,j]<cutoff}
                        remaining-=new; stack.extend(new)
                    groups.append(group)
                gap=min((float(dist[np.ix_(g,h)].min()) for i,g in enumerate(groups) for h in groups[i+1:]),default=None)
                components.append(dict(landing=index,cutoff_A=cutoff,sizes=sorted(map(len,groups)),minimum_intercomponent_distance_A=gap))
    rows.append(dict(arm=folder.name,stages=stages,rejected=sum(s['rejected'] for s in stages),components=components))
output=dict(scope='post-hoc geometry replay, two default native scalar guards only; no species flag, pressure scaling or native periodic parity claim',raw_run=str(run),stages=sum(len(r['stages']) for r in rows),rejected=sum(r['rejected'] for r in rows),rows=rows)
out.mkdir(exist_ok=False)
(out/'analysis.json').write_text(json.dumps(output,indent=2))
for row in rows: print(row['arm'],len(row['stages']),row['rejected'],max(s['maximum_atom_move_A'] for s in row['stages']),min(s['closest_ratio'] for s in row['stages']),row['components'])
