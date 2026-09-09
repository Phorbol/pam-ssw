"""Recovered fixed-cell LASP rotation primitives, independent of PAM strategies.

Static evidence: supplied GA-SSW ELF newssw_basics_mp_add_rotation_bias_
0x6e8360 (newssw_basics.F90:821,834-837), cbd_rotation_ 0x6e5590,
and rotate_dimer_ 0x6e57d0 (lines 712-715). No executable native oracle
has validated this reconstruction. These primitives do not implement the
native persistent Broyden history, angle control, retries or termination.
The native driver evaluates the requested endpoint and passes both forces
back; none of these functions calls a calculator. Scientific efficacy is
unvalidated. Coordinates are unwrapped Cartesian Angstrom, energies eV.
"""
from dataclasses import dataclass
import numpy as np


def _array(value, name):
    result=np.array(value,dtype=float,copy=True)
    if result.ndim!=2 or result.shape[1]!=3 or not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name} requires finite nonempty (N, 3) array')
    return result


def _direction(value):
    result=_array(value,'direction')
    if not np.isclose(np.linalg.norm(result),1.,rtol=1e-10,atol=1e-12):
        raise ValueError('direction must have unit Euclidean norm')
    return result


def _separation(value):
    value=float(value)
    if not np.isfinite(value) or value<=0:
        raise ValueError('separation must be finite and positive')
    return value


@dataclass(frozen=True)
class RotationQuadraticBias:
    """Rank-one Hessian shift: E=-weight*p²/2, F=weight*p*n.

    p=(R-center)·direction. ``weight`` is nonnegative eV/Angstrom²,
    a curvature shift, NOT the energy height of a deposited Gaussian.
    Native code adds only force; this energy is its analytic integral.
    There is no minimum-image remapping or stress/cell derivative.
    """
    center: np.ndarray
    direction: np.ndarray
    weight: float

    def __post_init__(self):
        center=_array(self.center,'center');direction=_direction(self.direction)
        weight=float(self.weight)
        if center.shape!=direction.shape:
            raise ValueError('center and direction shapes differ')
        if not np.isfinite(weight) or weight<0:
            raise ValueError('weight must be finite and nonnegative')
        center.setflags(write=False);direction.setflags(write=False)
        object.__setattr__(self,'center',center)
        object.__setattr__(self,'direction',direction)
        object.__setattr__(self,'weight',weight)

    def evaluate(self, atoms):
        positions=_array(atoms.positions,'positions')
        if positions.shape!=self.center.shape:
            raise ValueError('positions and center shapes differ')
        p=float(np.vdot(positions-self.center,self.direction))
        return -.5*self.weight*p*p,self.weight*p*self.direction


@dataclass(frozen=True)
class RotationObservation:
    curvature: float
    hessian_vector: np.ndarray
    tangent_force: np.ndarray


def rotation_observation(force_center, force_endpoint, direction, separation):
    """Native one-sided force geometry, with no calculator calls or mutation.

    Endpoint must be center + separation*direction on the SAME surface,
    including any rotation bias. Hn=(Fcenter-Fendpoint)/separation has
    first-order finite-difference truncation error. Curvature=n·Hn;
    tangent_force=-separation*(Hn-curvature*n), in eV/Angstrom. The
    native tangent force is deliberately NOT divided by separation.
    It is not the final reported native convergence norm, which also
    depends on the persistent FACT1 state and Broyden updates.
    """
    f0=_array(force_center,'force_center');f1=_array(force_endpoint,'force_endpoint')
    n=_direction(direction);dr=_separation(separation)
    if f0.shape!=n.shape or f1.shape!=n.shape:
        raise ValueError('forces and direction shapes differ')
    hv=(f0-f1)/dr
    curvature=float(np.vdot(hv,n))
    tangent=f1-f0+dr*curvature*n
    return RotationObservation(curvature,hv,tangent)


def dimer_endpoint(center, direction, separation):
    """Exact fixed-cell CBD endpoint request R1=R0+dr*n (source line 659).

    Caller owns fixed-coordinate projection and normalization; rejected
    invalid directions are not silently repaired. No variable-cell map.
    """
    center=_array(center,'center');direction=_direction(direction)
    if center.shape!=direction.shape:
        raise ValueError('center and direction shapes differ')
    return center+_separation(separation)*direction
