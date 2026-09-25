"""Post-hoc target geometry checks; no potential evaluations or search choices."""
import argparse
import json
from pathlib import Path
import networkx as nx
import numpy as np
from ase import Atoms
from ase.io import read

HERE=Path(__file__).resolve().parent

def graph(atoms):
    # Between the first and second reference neighbor shells; not a search bias.
    d=atoms.get_all_distances()
    g=nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    g.add_edges_from(zip(*np.where(np.triu((d>0)&(d<1.3*2.7),1))))
    return g

def geometry(atoms, reference):
    a=atoms.positions-atoms.positions.mean(axis=0)
    b=reference.positions-reference.positions.mean(axis=0)
    ga,gb=graph(atoms),graph(reference)
    matcher=nx.algorithms.isomorphism.GraphMatcher(ga,gb)
    best=None
    count=0
    for mapping in matcher.isomorphisms_iter():
        mapped=b[[mapping[i] for i in range(len(a))]]
        u,s,vt=np.linalg.svd(a.T@mapped)
        correction=np.diag([1.,1.,np.linalg.det(u@vt)])
        rms=float(np.sqrt(np.mean(np.sum((a@(u@correction@vt)-mapped)**2,axis=1))))
        best=rms if best is None else min(best,rms)
        count+=1
        if count>=1000: break
    return dict(graph_mappings_examined=count,proper_rms_A=best,
                connected_components=nx.number_connected_components(ga),
                geometry_match=bool(best is not None and best<0.03*2.7),
                interpretation='Positive RMS match proves geometric agreement; exhausted mapping cap without match is inconclusive.')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',type=Path)
    args=parser.parse_args()
    rows=[]
    for folder in sorted(args.runs.iterdir()):
        if not folder.is_dir(): continue
        for role in ('candidate','best'):
            path=folder/f'{role}.extxyz'
            if not path.exists(): continue
            atoms=read(path)
            n=len(atoms)
            reference=Atoms(f'Ar{n}',positions=2.7*np.loadtxt(HERE/'references'/f'lj{n}.points'))
            rows.append(dict(arm=folder.name,role=role,**geometry(atoms,reference)))
    target=args.runs/'geometry-analysis.json'
    target.write_text(json.dumps(dict(scope='Geometry only: combine with independent energy/force qualification',rows=rows),indent=2)+'\n')
    print(target)

if __name__=='__main__': main()
