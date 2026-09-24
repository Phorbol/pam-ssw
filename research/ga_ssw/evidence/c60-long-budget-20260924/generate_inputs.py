"""Frozen prospective random C60 inputs; no outcome-based replacement."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
root=Path(__file__).resolve().parent
for seed in (17101,17102):
    rng=np.random.default_rng(seed); accepted=[]; trials=0
    while len(accepted)<60:
        trials+=1
        if trials>100000:raise RuntimeError('initialization trial cap')
        x=rng.uniform(-5,5,3)
        if np.linalg.norm(x)>5:continue
        if accepted and np.min(np.linalg.norm(np.array(accepted)-x,axis=1))<1:continue
        accepted.append(x)
    positions=np.round(np.array(accepted)+25,10)
    atoms=Atoms('C60',positions=positions,cell=[50,50,50],pbc=False)
    write(root/f'seed{seed}.extxyz',atoms)
    (root/f'seed{seed}.json').write_text(json.dumps(dict(seed=seed,trials=trials,positions=positions.tolist()),indent=2)+'\n')
(root/'protocol.json').write_text(json.dumps(dict(category='prospective fixed protocol, two new inputs',seeds=[17101,17102],radius_A=5,minimum_separation_A=1,rule='uniform cube rejection into sphere, sequential pair-distance exclusion',parameters='Provisional finite-volume input distribution, retained from interface fixture; no optimality or literature-default claim',coordinate_rounding_decimals=10,translation_A=[25,25,25],pbc=False,acceptance='Report cage and same-model reference-energy metrics separately; no success-rate inference from two prospective seeds'),indent=2)+'\n')
