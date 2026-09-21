"""Raw element-resolved TYPE4 contact geometry; no radius classifier or PES."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.geometry import find_mic
from scipy.sparse.csgraph import minimum_spanning_tree


def describe(raw):
    a=Atoms(**raw);sup=np.arange(486);ads=np.arange(486,514);au=ads[a.numbers[ads]==79];oxy=ads[a.numbers[ads]==8]
    def distances(i,j):
        _,d=find_mic((a.positions[i,None]-a.positions[None,j]).reshape(-1,3),a.cell,a.pbc)
        return d.reshape(len(i),len(j))
    aa=distances(au,au);mst_max=float(minimum_spanning_tree(aa).data.max());np.fill_diagonal(aa,np.inf)
    result=dict(au_mst_longest_edge=mst_max,adsorbate_composition={a[i].symbol:int(sum(a.numbers[ads]==a.numbers[i])) for i in ads},
                au_nearest_au=dict(min=float(aa.min(axis=1).min()),max=float(aa.min(axis=1).max())),
                adsorbate_support_min=float(distances(ads,sup).min()),
                au_support_nearest_range=[float(x) for x in (distances(au,sup).min(axis=1).min(),distances(au,sup).min(axis=1).max())],oxygen=[])
    for i in oxy:
        other=oxy[oxy!=i]
        result['oxygen'].append(dict(index=int(i),nearest_Au=float(distances([i],au).min()),nearest_adsorbate_O=float(distances([i],other).min()),nearest_support_by_element={chemical_symbols[int(z)]:float(distances([i],sup[a.numbers[sup]==z]).min()) for z in np.unique(a.numbers[sup])}))
    return result


def main():
    base=Path(__file__).resolve().parent/'evidence';b=base/'type4-certified-start-control'
    raw={'initial':json.loads((b/'search/input.json').read_text())}
    for arm in ('physical_mask_only','source_direction_mask'):
        raw[arm]=json.loads((b/f'search/{arm}/result.json').read_text())['result']['minima'][-1]['atoms']
    output=dict(structures={k:describe(v) for k,v in raw.items()},scope='all distances Angstrom, minimum-image under source3D PBC; source support0:486,adsorbate486:514; raw contacts only, no bond thresholds, DFT or kinetic claim; zero PES')
    (b/'geometry-contacts.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
