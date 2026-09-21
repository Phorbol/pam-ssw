"""Independent isolated multi-chain RC-SSW; no native lambda or periodic map."""
from dataclasses import dataclass
import numpy as np
from .rc_reference import RCSSWConfig,_run_reduced_ssw
from .rc_forest import RigidForestChart,ForestSurface


@dataclass(frozen=True,kw_only=True)
class RCForestSSWConfig(RCSSWConfig):
    rotation_length: float  # Angstrom/radian for relative root rotations

    def __post_init__(self):
        super().__post_init__()
        if not np.isscalar(self.rotation_length) or not np.isfinite(self.rotation_length) or self.rotation_length<=0:
            raise ValueError('rotation_length must be positive finite')


def run_rc_forest_ssw(atoms,surface,*,trees,anchor,steps,config,rng):
    """Relative root poses + internal torsions, then bare Cartesian quench/MC.

    One nonlinear root pose fixes the isolated global gauge. Every other root
    retains its relative six coordinates. Explicit trees use global atom IDs.
    All true E/F, failures and rejected certified candidates use the shared
    single-chain lifecycle; chart references rebuild only after selection.
    """
    if not isinstance(config,RCForestSSWConfig):raise TypeError('RCForestSSWConfig required')
    if isinstance(steps,(bool,np.bool_)) or not isinstance(steps,(int,np.integer)) or steps<0:
        raise ValueError('nonnegative integer steps required')
    trees=tuple(dict(bodies=tuple(tuple(g) for g in t['bodies']),parents=tuple(t['parents']),joints=tuple(None if a is None else tuple(a) for a in t['joints'])) for t in trees)
    def factory(a):
        return ForestSurface(RigidForestChart(a,trees,anchor=anchor),surface,rotation_length=config.rotation_length,torsion_length=config.torsion_length)
    return _run_reduced_ssw(atoms,surface,steps=steps,config=config,rng=rng,factory=factory,coordinate_label="scaled_coordinates")
