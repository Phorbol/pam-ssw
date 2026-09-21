"""Isolated disjoint RC trees, fixing global pose once through an anchor root."""
import numpy as np
from .rc_geometry import RigidChainChart


class RigidForestChart:
    """Explicit trees: each dict supplies bodies/parents/joints in global indices.

    Anchor root must be nonlinear. All other root bodies must also be nonlinear
    to avoid redundant point/linear-body rotation coordinates. Coordinates are
    translations in Angstrom and finite root rotations/torsions in radians.
    """
    def __init__(self,atoms,trees,*,anchor=0):
        trees=tuple(trees)
        if not trees or isinstance(anchor,(bool,np.bool_)) or not isinstance(anchor,(int,np.integer)) or not 0<=anchor<len(trees):
            raise ValueError('valid explicit anchor tree required')
        if atoms.pbc.any() or atoms.constraints:raise ValueError('isolated unconstrained forest required')
        self.reference=atoms.copy();self.parts=[];seen=set();self.kinds=[]
        for k,tree in enumerate(trees):
            bodies=tuple(tuple(g) for g in tree['bodies'])
            ids=sorted(set(i for g in bodies for i in g))
            if not ids or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i<0 or i>=len(atoms) for i in ids):
                raise ValueError('invalid global atom indices')
            if seen.intersection(ids):raise ValueError('different trees must be disjoint')
            seen.update(ids);lookup={i:j for j,i in enumerate(ids)}
            joints=[None if a is None else tuple(lookup[i] for i in a) for a in tree['joints']]
            chart=RigidChainChart(atoms[ids],[[lookup[i] for i in g] for g in bodies],parents=tree['parents'],joints=joints)
            root=atoms.positions[list(bodies[0])];s=np.linalg.svd(root-root.mean(axis=0),compute_uv=False)
            if len(s)<2 or s[1]<=np.finfo(float).eps*max(root.shape)*s[0]:
                raise ValueError('all root bodies must be noncollinear')
            columns=np.arange(6 if k==anchor else 0,chart.dimension)
            start=len(self.kinds)
            self.kinds.extend('translation' if j<3 else 'rotation' if j<6 else 'torsion' for j in columns)
            self.parts.append((np.asarray(ids),chart,columns,slice(start,len(self.kinds))))
        if seen!=set(range(len(atoms))):raise ValueError('forest must cover all atoms')
        self.dimension=len(self.kinds);self.anchor=anchor

    def evaluate(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.dimension,) or not np.isfinite(q).all():raise ValueError('finite forest coordinate required')
        out=self.reference.copy();J=np.zeros((len(out),3,self.dimension))
        for ids,chart,columns,span in self.parts:
            local=np.zeros(chart.dimension);local[columns]=q[span]
            a,j=chart.evaluate(local);out.positions[ids]=a.positions
            J[ids,:,span]=j[:,:,columns]
        return out,J

    def pullback(self,q,forces):
        f=np.asarray(forces,dtype=float)
        if f.shape!=(len(self.reference),3) or not np.isfinite(f).all():raise ValueError('finite Cartesian forces required')
        return np.einsum('ijk,ij->k',self.evaluate(q)[1],f)


class ForestSurface:
    """Exact energy/gradient in explicitly scaled relative-pose/torsion space."""
    def __init__(self,chart,surface,*,rotation_length,torsion_length):
        if any(not np.isfinite(x) or x<=0 for x in (rotation_length,torsion_length)):
            raise ValueError('positive finite angular lengths required')
        self.chart=chart;self.surface=surface;self.dimension=chart.dimension
        if self.dimension<1:raise ValueError('forest has no internal or relative degrees of freedom')
        self.scales=np.array([1. if k=='translation' else rotation_length if k=='rotation' else torsion_length for k in chart.kinds])

    def coordinates(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(self.dimension,) or not np.isfinite(x).all():raise ValueError('finite scaled forest coordinate required')
        coordinates=x/self.scales
        from .rc_optimization_domain import check_principal_root_rotations
        check_principal_root_rotations(coordinates,self.chart.kinds)
        return coordinates

    def atoms(self,x):return self.chart.evaluate(self.coordinates(x))[0]

    def evaluate(self,x):
        a,J=self.chart.evaluate(self.coordinates(x));e,f=self.surface.evaluate(a);f=np.asarray(f,dtype=float)
        if not np.isfinite(e) or f.shape!=a.positions.shape or not np.isfinite(f).all():raise ValueError('invalid physical E/F')
        return float(e),-np.einsum('ijk,ij->k',J,f)/self.scales
