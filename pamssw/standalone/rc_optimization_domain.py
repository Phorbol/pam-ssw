"""Nonsingular root-rotation domain for RC optimization, not raw geometry."""
import math
import numpy as np
from .rc_vc_geometry import RigidForestCellChart


class RootRotationChartError(ValueError):
    """Trial leaves the open principal SO(3) logarithm ball."""


def check_principal_root_rotations(coordinates,kinds):
    """Require ||omega||<pi for each root, in actual radians after unscaling.

    The pi boundary is the principal-log cut, not an adjustable search radius.
    Torsions remain unwrapped. No normalization, wrapping or history change.
    """
    q=np.asarray(coordinates,dtype=float)
    if q.shape!=(len(kinds),) or not np.isfinite(q).all():raise ValueError('finite pose coordinates required')
    indices=np.array([i for i,k in enumerate(kinds) if k=='rotation'],dtype=int)
    if len(indices)%3:raise ValueError('root rotation blocks must contain three components')
    for number,block in enumerate(indices.reshape(-1,3)):
        if np.linalg.norm(q[block])>=math.pi:
            raise RootRotationChartError(f'root rotation {number} outside open principal log ball (norm must be < pi)')


class PrincipalRigidForestCellChart(RigidForestCellChart):
    """Optimizer view; base raw geometry still supports arbitrary finite angles."""
    def geometry(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.dimension,) or not np.isfinite(q).all():raise ValueError('finite full optimization coordinate required')
        check_principal_root_rotations(q[:-6]/self.scales,self.kinds)
        return super().geometry(q)
