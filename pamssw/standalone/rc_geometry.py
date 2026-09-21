"""Exact finite-rotation articulated tree geometry, an independent RC subset.

Root translation/rotation vector plus one torsion per shared bond. This is not
native coordinate-gauge parity, lambda force transmission, variable-cell RC,
or an RC-SSW driver. Generalized forces are the exact work-conjugate J.T F.
"""
import numpy as np
from scipy.linalg import expm, expm_frechet


def _skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


class RigidChainChart:
    """A connected rooted tree with two shared atoms on every parent-child edge.

    parents must be topologically ordered, root parent=-1. joints[j]=(a,b)
    gives the oriented reference axis a->b in parent and child. Membership of
    any atom must be a connected subtree, preventing inconsistent shared atoms.
    q[:3] is root translation in Angstrom; q[3:6] is rotation vector in radians;
    q[6:] are bond torsions in radians. No metric or dimensional scaling is
    implicit. Fixed unwrapped reference, isolated atoms, no ASE constraints.
    """
    def __init__(self, atoms, bodies, *, parents, joints):
        if atoms.pbc.any() or atoms.constraints:
            raise ValueError('only nonperiodic unconstrained reference supported')
        self.reference=atoms.copy()
        self.bodies=tuple(tuple(g) for g in bodies)
        self.parents=tuple(parents);self.joints=tuple(joints)
        n=len(atoms);m=len(self.bodies)
        if not n or not m or len(parents)!=m or len(joints)!=m or parents[0]!=-1 or joints[0] is not None:
            raise ValueError('one rooted topologically ordered tree required')
        if not np.isfinite(atoms.positions).all():raise ValueError('finite reference required')
        sets=[]
        for j,g in enumerate(self.bodies):
            if not g or len(set(g))!=len(g) or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or i<0 or i>=n for i in g):
                raise ValueError('invalid body atom indices')
            sets.append(set(g))
            if j:
                p=parents[j]
                if isinstance(p,bool) or not isinstance(p,(int,np.integer)) or p<0 or p>=j:
                    raise ValueError('parents must precede children')
                axis=joints[j]
                if axis is None or len(axis)!=2 or axis[0]==axis[1] or set(axis)!=sets[p]&sets[j]:
                    raise ValueError('parent-child overlap must be exactly the two joint atoms')
        if set.union(*sets)!=set(range(n)):raise ValueError('bodies must cover every atom')
        for i in range(n):
            members=[j for j,g in enumerate(sets) if i in g]
            edges=sum(j>0 and parents[j] in members for j in members)
            if edges!=len(members)-1:raise ValueError('shared atom membership must be connected')
        # A bond torsion must move something beyond its two invariant axis
        # endpoints. An axis-only whole subtree creates an identically zero
        # Jacobian column, not an additional physical search freedom.
        for j in range(1,m):
            descendants={j}
            for k in range(j+1,m):
                if parents[k] in descendants:descendants.add(k)
            moved=set().union(*(sets[k] for k in descendants))
            if moved <= set(joints[j]):
                raise ValueError('axis-only descendant subtree has no torsional degree of freedom')
        self.dimension=6+m-1
        self._positions=atoms.positions.copy()
        self._pivot=self._positions[list(self.bodies[0])].mean(axis=0)
        self._owners=np.array([next(j for j,g in enumerate(sets) if i in g) for i in range(n)])
        self._generators=[None]
        for a,b in joints[1:]:
            delta=self._positions[b]-self._positions[a];length=np.linalg.norm(delta)
            if length<=0:raise ValueError('joint axis has zero length')
            W=_skew(delta/length);G=np.zeros((4,4));G[:3,:3]=W;G[:3,3]=-W@self._positions[a]
            self._generators.append(G)

    def evaluate(self,q):
        """Return independent Atoms and exact Cartesian Jacobian [N,3,ndof].

        Root exp derivatives use Frechet derivatives, including finite angles.
        A child transform is T_parent exp(theta*G_reference_axis). This chooses
        a transported-parent torsion gauge, not the paper's shortest-alignment
        gauge. Every shared axis point is invariant under its local transform.
        """
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.dimension,) or not np.isfinite(q).all():raise ValueError('finite q of chart dimension required')
        W=_skew(q[3:6]);R=expm(W);T=np.eye(4);T[:3,:3]=R
        T[:3,3]=q[:3]+self._pivot-R@self._pivot
        dT=np.zeros((self.dimension,4,4))
        for k in range(3):
            dT[k,k,3]=1.
            dR=expm_frechet(W,_skew(np.eye(3)[k]),compute_expm=False)
            dT[3+k,:3,:3]=dR;dT[3+k,:3,3]=-dR@self._pivot
        transforms=[T];derivatives=[dT]
        for j in range(1,len(self.bodies)):
            p=self.parents[j];G=self._generators[j];A=expm(q[5+j]*G)
            transforms.append(transforms[p]@A)
            D=derivatives[p]@A;D[5+j]+=transforms[p]@G@A;derivatives.append(D)
        positions=np.empty_like(self._positions);J=np.empty((len(positions),3,self.dimension))
        for i,owner in enumerate(self._owners):
            v=np.r_[self._positions[i],1.]
            positions[i]=(transforms[owner]@v)[:3]
            J[i]=(derivatives[owner]@v)[:,:3].T
        out=self.reference.copy();out.positions=positions
        return out,J

    def pullback(self,q,forces):
        """Work-conjugate forces: translation eV/A, angles eV/radian."""
        f=np.asarray(forces,dtype=float)
        if f.shape!=self._positions.shape or not np.isfinite(f).all():raise ValueError('finite Cartesian forces required')
        _,J=self.evaluate(q)
        return np.einsum('ijk,ij->k',J,f)
