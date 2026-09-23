"""Read saved GA lineage and selected seeds; no calculator calls."""
import json
import pickle
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
for seed in (3,17):
    folder=root/'runs'/f'c60-seed{seed}-ga-v2'
    with (folder/'result.pkl').open('rb') as f:r=pickle.load(f)
    by_id={o.id:o for o in r.observations}
    def observation(o):
        return dict(id=o.id,phase=o.phase,energy=o.result.energy,qualified=o.eligible_for_archive,
                    operator=o.operator,parent_ids=list(set(o.parent_ids)),details=o.details)
    children=[o for o in r.observations if o.phase=='offspring_quench']
    selected=[dict(phase=s.phase,seed_id=s.seed_id,observation=observation(by_id[s.seed_id]))
              for s in r.stages if s.phase in ('generation_short','fine') and s.seed_id is not None]
    quick=[o for o in r.observations if o.phase=='quick' and o.eligible_for_archive]
    rows.append(dict(seed=seed,best_quick=observation(min(quick,key=lambda o:o.result.energy)),
                     child_operators=dict(Counter(o.operator for o in children)),
                     children=[observation(o) for o in children],selected=selected))
def encode(value):
    if hasattr(value,'tolist'):return value.tolist()
    raise TypeError(type(value).__name__)
(root/'lineage-readout.json').write_text(json.dumps(rows,indent=2,default=encode)+'\n')
for r in rows:
 print('seed',r['seed'],'best_quick',r['best_quick']['id'],r['best_quick']['energy'])
 print('child_ops',r['child_operators'])
 for c in r['children']:print('child',c['id'],c['operator'],c['energy'],c['qualified'],c['parent_ids'])
 for s in r['selected']:print('selected',s['phase'],s['seed_id'],s['observation']['phase'],s['observation']['operator'],s['observation']['energy'])
