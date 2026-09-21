"""Unconstrained fixed-cell control recovered from rotate_dimer ELF.

See docs/research/native-rotation-followup.md for instruction addresses. These
pure state transitions do not replace BRZERO4, evaluate a PES, or constitute a
complete native rotation. Constants below are executable facts, not new knobs.
"""
from dataclasses import dataclass
import math
from numbers import Integral
import numpy as np
from .native_rotation import _array, _direction


def _count(value, name, minimum):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum:
        raise ValueError(f'{name} must be integer >= {minimum}')
    return int(value)


def cap_rotation(evaluated_direction, candidate):
    """Normalize candidate and apply native 40-degree cap in its tangent plane.

    No fixed-coordinate mask, molecular constraints or cell map is implemented.
    Zero vectors, acos-domain violations and antiparallel zero tangents fail
    explicitly rather than inventing a direction. Inputs are not mutated.
    """
    old=_direction(evaluated_direction)
    new=_array(candidate,'candidate')
    if old.shape!=new.shape:
        raise ValueError('direction shapes differ')
    length=float(np.linalg.norm(new))
    if not math.isfinite(length) or length==0:
        raise ValueError('candidate must have finite nonzero norm')
    new/=length
    dot=float(np.vdot(old,new))
    if not -1<=dot<=1:
        raise ValueError('native acos outside numerical domain')
    if math.acos(dot)*180./math.pi>40.:
        tangent=new-dot*old
        length=float(np.linalg.norm(tangent))
        if length==0:
            raise ValueError('zero tangent outside recovered finite domain')
        new=old+0.83909963117727993*tangent/length
        new/=np.linalg.norm(new)
    return new


def retry_factor(rotnum, attempt, candidate_norm, factor1):
    """Return reduced FACT1, or None when original code does not retry.

    Call after an update, before normalization. On retry the caller restores
    n00, undoes prior force scaling and reinitializes BRZERO4; no new PES call.
    Original FACT argument must remain distinct from this evolving FACT1.
    """
    rotnum=_count(rotnum,'rotnum',1);attempt=_count(attempt,'attempt',1)
    if not math.isfinite(candidate_norm) or candidate_norm<0:
        raise ValueError('candidate norm must be finite and nonnegative')
    if not math.isfinite(factor1) or factor1<=0:
        raise ValueError('FACT1 must be finite and positive')
    if rotnum==1 and candidate_norm>1.02 and attempt<15:
        reduced=factor1*.8
        if reduced==0:
            raise ValueError('FACT1 underflow outside recovered domain')
        return reduced
    return None


@dataclass(frozen=True)
class RotationFinish:
    direction: np.ndarray
    next_rotnum: int
    stop: bool
    force_converged: bool
    budget_exceeded: bool
    prerot_override: bool


def finish_rotation(evaluated_direction, candidate_direction, *, rotnum, rotmax,
                    reported_force, ftol, infor, curv_real):
    """Native strict predicates and terminal n00 rollback, after angular cap.

    reported_force is the native workspace norm/FACT1*10, NOT an HVP residual.
    Caller supplies curv_real from the native real-surface calculation. The
    CBD_PreRot exception can override rotmax; an external hard request budget
    remains necessary and is not native convergence. Fortran trailing blanks
    are ignored, but other whitespace is not silently stripped.
    """
    old=_direction(evaluated_direction);new=_direction(candidate_direction)
    if old.shape!=new.shape:
        raise ValueError('direction shapes differ')
    rotnum=_count(rotnum,'rotnum',1);rotmax=_count(rotmax,'rotmax',0)
    if not all(math.isfinite(x) for x in (reported_force,ftol,curv_real)) or min(reported_force,ftol)<0:
        raise ValueError('finite nonnegative force/tolerance and finite curvature required')
    if not isinstance(infor,str):
        raise TypeError('infor must be a Fortran-style string')
    force=reported_force<ftol;budget=rotnum>rotmax
    override=infor.rstrip(' ')=='CBD_PreRot' and curv_real< -1e-6
    stop=(force or budget) and not override
    direction=(old if stop else new).copy();direction.setflags(write=False)
    return RotationFinish(direction,rotnum+1,bool(stop),bool(force),bool(budget),bool(override))
