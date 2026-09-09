"""Conservative rigid/species-permutation alignment of real archived structures.

Returned RMSD is a witnessed alignment, not a global minimum or basin identity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from validate_water_archive import read_arc


def witnessed_rmsd(za,a,zb,b):
    a=a-a.mean(axis=0);b=b-b.mean(axis=0)
    def fingerprint(z,x):
        distances=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
        return np.concatenate([np.sort(distances[:,z==s],axis=1) for s in sorted(set(z))],axis=1)
    fa=fingerprint(za,a);fb=fingerprint(zb,b)
    cost=np.linalg.norm(fa[:,None,:]-fb[None,:,:],axis=2);cost[za[:,None]!=zb[None,:]]=np.inf
    _,assignment=linear_sum_assignment(cost);matched=b[assignment]
    u,_,vt=np.linalg.svd(matched.T@a);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
    aligned=matched@(u@d@vt)
    return float(np.sqrt(np.mean(np.sum((aligned-a)**2,axis=1))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    structures=[]
    for arc in sorted(args.archive.glob('*.arc'),key=lambda p:int(p.stem)):
        _,_,symbols,xyz,e=read_arc(arc);structures.append((arc.stem,np.array(symbols),xyz,e))
    pairs=[]
    for i,(name,z,xyz,e) in enumerate(structures):
        for other,z2,xyz2,e2 in structures[i+1:]:
            pairs.append(dict(a=name,b=other,witnessed_rmsd=witnessed_rmsd(z,xyz,z2,xyz2),energy_difference=abs(e-e2)))
    pairs.sort(key=lambda p:p['witnessed_rmsd'])
    result=dict(structures=len(structures),pairs=pairs,near_pair_counts={str(t):sum(p['witnessed_rmsd']<=t for p in pairs) for t in [1e-5,1e-4,1e-3,.01]},
                method='Species-restricted Hungarian matching of sorted distance fingerprints, then proper Kabsch rotation; witnessed upper-bound RMSD. No basin-count claim.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v[:6] if k=='pairs' else v for k,v in result.items()},indent=2))


if __name__=='__main__':main()
