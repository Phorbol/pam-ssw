"""Pure C60 graph validator copied from the plan's validator_source."""
from collections import Counter
import networkx as nx
import numpy as np

def graph_row(numbers,positions,cutoff,reference_graph=None):
    x=np.asarray(positions,float); g=nx.Graph(); g.add_nodes_from(range(len(x)))
    d=np.linalg.norm(x[:,None]-x[None,:],axis=2)
    g.add_edges_from(zip(*np.where(np.triu((d<cutoff)&(d>0),1))))
    planar,emb=nx.check_planarity(g); faces=[]
    if planar:
        seen=set()
        for u,v in emb.edges():
            if (u,v) not in seen: faces.append(emb.traverse_face(u,v,seen))
    degrees=dict(g.degree()); deg=Counter(degrees.values())
    three_connected=bool(len(g)>=3 and nx.node_connectivity(g)>=3) if nx.is_connected(g) else False
    cage=bool(len(x)==60 and all(z==6 for z in numbers) and nx.is_connected(g) and len(g.edges())==90
              and all(v==3 for v in degrees.values()) and three_connected and planar
              and Counter(map(len,faces))==Counter({5:12,6:20}))
    ih=bool(reference_graph is not None and nx.is_isomorphic(g,reference_graph))
    return dict(cutoff_A=cutoff,all_carbon=bool(len(x)==60 and all(z==6 for z in numbers)),
                components=nx.number_connected_components(g),edges=g.number_of_edges(),degree_counts=dict(deg),
                three_connected=three_connected,planar=planar,face_counts=dict(Counter(map(len,faces))),
                graph_cage_candidate=cage,ih_graph_match=ih)

def conditional_candidate(minima):
    """First converged 1.8-A cage not already covered by initial or best."""
    if not minima:return None,[]
    best_index=min(range(len(minima)),key=lambda i:minima[i].energy)
    diagnostics=[]
    for index,minimum in enumerate(minima):
        rows={str(c):graph_row(minimum.atoms.numbers,minimum.atoms.positions,c) for c in (1.8,1.64,1.7)}
        diagnostics.append(dict(index=index,converged=bool(minimum.converged),graphs=rows))
        if minimum.converged and rows['1.8']['graph_cage_candidate']:
            return (None if index in (0,best_index) else index),diagnostics
    return None,diagnostics
