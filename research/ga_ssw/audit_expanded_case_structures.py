"""Offline geometry diagnostics; no calculator or search-state mutations."""
import argparse
import json
from pathlib import Path
import numpy as np
import networkx as nx
from ase import Atoms
from ase.collections import g2
from pamssw.standalone.native_ls import HCO_BOND_LENGTHS


def atoms_from(d):
    return Atoms(numbers=d['numbers'], positions=d['positions'], cell=d['cell'], pbc=d['pbc'])


def graph(a):
    g=nx.Graph()
    for i,z in enumerate(a.numbers):g.add_node(i,Z=int(z))
    distances=a.get_all_distances(mic=False)
    for i in range(len(a)):
        for j in range(i):
            z=(int(a.numbers[i]),int(a.numbers[j]))
            length=HCO_BOND_LENGTHS.get(z,HCO_BOND_LENGTHS.get(z[::-1]))
            if length is not None and distances[i,j] < length+.1:g.add_edge(i,j)
    return g


def describe(a):
    distance=a.get_all_distances(mic=False)
    pairs=distance[np.triu_indices(len(a),1)]
    near=distance+np.eye(len(a))*1e100
    out=dict(diameter=float(pairs.max()),nearest_min=float(near.min()),
             nearest_max=float(near.min(axis=1).max()),composition=a.get_chemical_formula())
    if set(a.numbers).issubset({1,6,8}):
        g=graph(a);components=list(nx.connected_components(g))
        out['component_compositions']=[a[sorted(c)].get_chemical_formula() for c in components]
        out['connected']=len(components)==1
        if a.get_chemical_formula()=='C4H6':
            match=lambda x,y:x['Z']==y['Z']
            out['reference_connectivity_matches']=[name for name in
                ('bicyclobutane','methylenecyclopropane','cyclobutene','butadiene','2-butyne')
                if nx.is_isomorphic(g,graph(g2[name]),node_match=match)]
        if a.get_chemical_formula()=='H30O15':
            out['fifteen_intact_waters']=len(components)==15 and all(x=='H2O' for x in out['component_compositions'])
            oxygen=np.flatnonzero(a.numbers==8)
            oo=distance[np.ix_(oxygen,oxygen)]
            out['sorted_oxygen_pair_distances']=np.sort(oo[np.triu_indices(len(oxygen),1)]).tolist()
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    rows=[]
    for path in sorted(args.directory.rglob('result.json')):
        if 'source' in path.parts:continue
        result=json.loads(path.read_text())
        if not isinstance(result,dict) or 'minima' not in result:continue
        initial=result.get('initial') or {}
        initial_e=initial.get('energy')
        minima=[]
        for i,m in enumerate(result['minima']):
            a=atoms_from(m['atoms']);minima.append(dict(index=i,energy=m['energy'],
                delta_initial=None if initial_e is None else m['energy']-initial_e,
                stored_fmax=m.get('max_force'),**describe(a)))
        rows.append(dict(arm=str(path.parent.relative_to(args.directory)),status=result.get('status'),
            attempts=len(result.get('records',[])),accepted=sum(bool(r.get('accepted')) for r in result.get('records',[])),
            record_statuses=[r.get('status') for r in result.get('records',[])],minima=minima))
    output=args.directory/'root-structure-audit.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(dict(rows=rows,limits='Offline geometry only. HCO table length + .1 Angstrom graph; no bond order, stereochemical or exact basin certification. No new PES requests.'),indent=2)+'\n')
    print(json.dumps(dict(arms=len(rows),observations=sum(len(r['minima']) for r in rows))))

if __name__=='__main__':main()
