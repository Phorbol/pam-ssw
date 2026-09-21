"""Offline Cu13 identity-view demonstration; no calculator/PES calls."""
import argparse, json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.minimum_identity import MinimumIdentityView, update_identity_view

SOURCE = Path('research/ga_ssw/evidence/cu13-safe-total/strict-validation/17-dimer.json')

def pair_distance_comparator(a, b):
    """Caller approximation: sorted all pair distances, no default metric."""
    if tuple(a.numbers) != tuple(b.numbers): return False
    da=np.sort(a.get_all_distances(mic=False).ravel()); db=np.sort(b.get_all_distances(mic=False).ravel())
    return bool(np.allclose(da, db, atol=1e-6, rtol=0.0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=json.loads(SOURCE.read_text()); selected=[]
    for row in rows:
        if row['qualified'] and (not selected or row['fingerprint_group'] != selected[0]['fingerprint_group']): selected.append(row)
        if len(selected)==2: break
    if len(selected)!=2: raise RuntimeError('fewer than two qualified groups in source')
    a=Atoms('Cu13',positions=selected[0]['positions']); b=Atoms('Cu13',positions=selected[1]['positions'])
    t=.37; rot=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
    ae=a.copy(); ae.positions=a.positions@rot.T+[3.,-2.,1.]
    be=b.copy(); be.positions=b.positions[::-1]@rot.T+[-1.,2.,-.5]
    minima=[type('Obs',(),{'atoms':x})() for x in (a,ae,b,be)]
    view=update_identity_view(MinimumIdentityView([],[],[],0,0),minima,pair_distance_comparator)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'inputs.json').write_text(json.dumps(dict(source=str(SOURCE),selected=[{k:r[k] for k in ('index','qualified','fingerprint_group','energy','positions')} for r in selected],observation_count=4,comparator='sorted all pair distances, atol=1e-6',transformations=['identity','rigid rotation+translation','different qualified structure','permutation+rigid rotation+translation']),indent=2)+'\n')
    (args.output/'result.json').write_text(json.dumps(dict(representative_indices=view.representative_indices,observation_to_representative=view.observation_to_representative,failures=view.failures,match_calls=view.match_calls,processed_count=view.processed_count,raw_observations=4,limitation='caller approximation; does not prove general identity or same chirality'),indent=2)+'\n')
if __name__=='__main__': main()
