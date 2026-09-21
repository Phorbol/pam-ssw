"""Experimental unweighted linear Eckart section for isolated free clusters.

This is a coordinate restriction, not native LASP parity or a universal ASE
constraint. The caller declares translation/rotation invariance of its PES.
The fixed Euclidean metric matches this SSW's Cartesian displacement metric.
See docs/research/cluster-eckart-design-review.md for derivation and limits.
"""
import numpy as np
from ase.filters import Filter


class ClusterFrameDomainError(ValueError):
    """A trial left the regular positive alignment chart."""


class ClusterFrame:
    """X=X0+P z; P removes the six rigid tangent vectors at fixed X0.

    An affine section is local: the positive Eckart alignment branch must stay
    regular. Reject branch loss instead of silently rotating coordinates while
    leaving Gaussian centers/directions or optimizer history unchanged.
    """
    def __init__(self, atoms):
        if len(atoms)<3 or atoms.pbc.any() or atoms.constraints:
            raise ValueError('cluster frame requires >=3 unconstrained nonperiodic atoms')
        self.reference=atoms.positions.copy()
        if not np.isfinite(self.reference).all():
            raise ValueError('finite cluster coordinates required')
        self.relative=self.reference-self.reference.mean(axis=0)
        rotational=np.column_stack([np.cross(axis,self.relative).ravel() for axis in np.eye(3)])
        u,s,_=np.linalg.svd(rotational,full_matrices=False)
        tol=np.finfo(float).eps*max(rotational.shape)*s[0]
        if np.count_nonzero(s>tol)!=3:
            raise ValueError('linear or coincident reference has no six-mode regular frame')
        translation=np.column_stack([np.broadcast_to(axis,self.relative.shape).ravel()/np.sqrt(len(atoms)) for axis in np.eye(3)])
        self.basis=np.column_stack([translation,u])
        self.reference.setflags(write=False)
        self.relative.setflags(write=False)
        self.basis.setflags(write=False)

    def project(self, vector):
        v=np.asarray(vector,dtype=float)
        if v.shape!=self.reference.shape or not np.isfinite(v).all():
            raise ValueError('finite frame-shaped vector required')
        flat=v.ravel()
        return (flat-self.basis@(self.basis.T@flat)).reshape(v.shape)

    def positions(self, candidate):
        x=self.reference+self.project(np.asarray(candidate)-self.reference)
        relative=x-x.mean(axis=0)
        cross=self.relative.T@relative
        alignment=np.trace(cross)*np.eye(3)-cross
        alignment=(alignment+alignment.T)/2
        eigen=np.linalg.eigvalsh(alignment)
        # Numerical rank/branch check only, not a physical search parameter.
        tol=np.finfo(float).eps*max(self.reference.size,3)*max(abs(eigen))
        if eigen[0]<=tol:
            raise ClusterFrameDomainError('cluster frame left its regular positive alignment branch')
        return x


class ClusterFrameFilter(Filter):
    """Exact pullback of the COMPLETE calculator objective on a fixed section.

    Every trial coordinate is X0+P(z-X0), and every gradient is P times the
    complete gradient at that coordinate. Energy is never replaced or rescaled.
    Redundant normal coordinates are harmless zero directions; no dense 3N²
    projector or full null-space basis is stored.
    """
    def __init__(self, atoms, frame):
        super().__init__(atoms,indices=range(len(atoms)))
        self.frame=frame
        self.set_positions(atoms.positions)

    def set_positions(self, positions, **kwargs):
        self.atoms.set_positions(self.frame.positions(positions),**kwargs)

    def get_forces(self, *args, **kwargs):
        return self.frame.project(self.atoms.get_forces(*args,**kwargs))
