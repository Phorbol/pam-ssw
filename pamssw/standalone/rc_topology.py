"""Explicit LASP rigidbody/blist input mapped to an independent rooted forest.

Format: RC-SSW SI (2025), section7. One-based source indices become zero-based.
Roots and axis signs below are deterministic independent coordinate conventions,
not a recovered LASP central-body or inverse-coordinate gauge.
"""
from dataclasses import dataclass
from pathlib import Path
from collections import deque


@dataclass(frozen=True)
class RigidTopology:
    components: tuple
    bonds: tuple
    natoms: int


def _lines(path):
    return [v for line in Path(path).read_text().splitlines() if (v:=line.split('#',1)[0].strip())]


def read_rigid_topology(rigidbody_path,blist_path,*,natoms):
    if isinstance(natoms,bool) or not isinstance(natoms,int) or natoms<1:raise ValueError('positive natoms required')
    tokens=[int(v) for line in _lines(rigidbody_path) for v in line.split()]
    if not tokens or tokens[0]<1:raise ValueError('positive body count required')
    count=tokens[0];cursor=1;groups=[]
    for _ in range(count):
        if cursor>=len(tokens):raise ValueError('truncated rigidbody count')
        n=tokens[cursor];cursor+=1
        if n<1 or cursor+n>len(tokens):raise ValueError('invalid/truncated body membership')
        group=tuple(i-1 for i in tokens[cursor:cursor+n]);cursor+=n
        if len(set(group))!=n or any(i<0 or i>=natoms for i in group):raise ValueError('invalid/duplicate body atom')
        groups.append(group)
    if cursor!=len(tokens):raise ValueError('trailing rigidbody tokens')
    if set(i for g in groups for i in g)!=set(range(natoms)):raise ValueError('bodies must cover all atoms')
    bonds=set()
    for line in _lines(blist_path):
        row=tuple(int(v)-1 for v in line.split())
        if len(row)!=2 or row[0]==row[1] or any(i<0 or i>=natoms for i in row):raise ValueError('blist needs two distinct valid atom indices per line')
        key=tuple(sorted(row))
        if key in bonds:raise ValueError('duplicate blist bond')
        bonds.add(key)
    neighbors=[[] for _ in groups];axes={};sets=list(map(set,groups))
    for i,left in enumerate(sets):
        for j in range(i+1,count):
            common=left&sets[j]
            if len(common)>2:raise ValueError('body overlap larger than one joint axis')
            if len(common)==2:
                axis=tuple(sorted(common))
                if axis not in bonds:raise ValueError('shared joint axis missing from blist')
                neighbors[i].append(j);neighbors[j].append(i);axes[i,j]=axes[j,i]=axis
    seen=set();components=[]
    for root in range(count):
        if root in seen:continue
        queue=deque([(root,None)]);order=[];parents=[];joints=[];local={}
        while queue:
            node,parent=queue.popleft()
            if node in seen:raise ValueError('body graph contains a loop; tree coordinates cannot close it')
            seen.add(node);local[node]=len(order);order.append(node)
            parents.append(-1 if parent is None else local[parent]);joints.append(None if parent is None else axes[node,parent])
            for child in sorted(neighbors[node]):
                if child!=parent:queue.append((child,node))
        component_atoms=set.union(*(sets[i] for i in order))
        for atom in component_atoms:
            members={i for i in order if atom in sets[i]}
            edges=sum(parents[k]>=0 and order[parents[k]] in members for k,i in enumerate(order) if i in members)
            if edges!=len(members)-1:raise ValueError('shared atom membership is not connected along body tree')
        if any(component_atoms & set(i for g in c['bodies'] for i in g) for c in components):
            raise ValueError('separate chains overlap without a shared joint axis')
        components.append(dict(bodies=tuple(groups[i] for i in order),parents=tuple(parents),joints=tuple(joints),source_body_indices=tuple(order)))
    if any(not any(set(bond)<=g for g in sets) for bond in bonds):raise ValueError('blist bond not represented by any rigid body')
    return RigidTopology(tuple(components),tuple(sorted(bonds)),natoms)
