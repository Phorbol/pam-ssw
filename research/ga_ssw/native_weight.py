"""Finite-domain static reconstruction of LASP ``set_thisgaussw``.

Provenance: uploaded GA-SSW ELF, newssw_basics_mp_set_thisgaussw_ at
0x6e1730, DWARF newssw_basics.F90:166-203. Parameter locations are in
analysis/selected-dwarf.txt:177704-177815 of the external research archive;
0x6e1cba-0x6e1cf7 computes the angle, 0x6e1fce-0x6e1ffe updates weight,
and 0x6e257b-0x6e258a checks maxw before repeating. See external
analysis/ssw-kernel-comparison.md and kernel-static-probes.txt.

This is NOT executable-oracle-verified parity. A bounded GDB probe on
2026-09-09 failed before main: ``ptrace: Operation not permitted``. No
native helper or PES evaluation ran. NumPy reduction order may differ
from the native vectorized loops; threshold-adjacent bitwise parity is
not asserted. This module is not connected to the production walker.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class NativeWeightResult:
    weight: float
    energy: float
    force: np.ndarray
    angle_degrees: float
    updates: int
    stop_reason: str
    evidence: str = 'static_reconstruction_not_native_oracle_verified'


def adjust_native_weight(*, fa0, fa2, n, d1, d2, e2, w, maxw, step, scalefact0):
    """Adjust the last Gaussian using the recovered native operand contract.

    ``fa0`` is the incoming force accumulator, ``fa2`` the force array
    subtracted at entry, and ``n`` the caller-normalized Gaussian direction
    (all shape (N, 3)). Entry computes ``fa0 - fa2 + d1*w*d2*n``. These are
    the original argument names; callers must supply that accumulator and
    subtracted contribution, not assume ``fa0`` is always bare PES force.
    Their complete caller-side lifetime is not reconstructed here.

    The native addgaussian call site identifies ``d1=exp(-p*p/(2*sigma²))``
    and ``d2=p/sigma²`` where ``p=(R-Rcenter)·n``. ``e2`` is the energy
    accumulator excluding the adjustable term: output energy is e2+d1*w.
    W and e2 use the caller's energy unit; d2 inverse length; the literal
    additive ceiling 2 uses that same native energy unit (no conversion
    or universal-default claim). n, d1, step and scalefact0 are dimensionless.

    Supported domain: finite nonempty arrays, unit n, 0<d1<=1, d2>0,
    w>0, maxw>=0, scalefact0>1 and step>=1. Every resultant force must
    remain finite/nonzero and each weight update must make representable
    progress. Degenerate/nonunit directions, zero/negative projection,
    underflowed Gaussian, invalid acos inputs and non-growing schedules
    are rejected, not assigned invented native fallbacks. No normalization,
    angle clipping, or maxw clipping is performed. No inputs are mutated.
    """
    force=np.array(fa0,dtype=float,copy=True)
    subtract=np.asarray(fa2,dtype=float)
    direction=np.asarray(n,dtype=float)
    if force.ndim!=2 or force.shape[1]!=3 or not force.size:
        raise ValueError('force arrays require nonempty shape (N, 3)')
    if subtract.shape!=force.shape or direction.shape!=force.shape:
        raise ValueError('fa0, fa2 and n must have identical shapes')
    if not all(np.isfinite(a).all() for a in (force,subtract,direction)):
        raise ValueError('array operands must be finite')
    # Validation tolerance only, not normalization or a native algorithm setting.
    if abs(float(np.linalg.norm(direction))-1.)>1e-12:
        raise ValueError('n must already be a unit vector')
    d1,d2,e2,w,maxw,step,scale=map(float,(d1,d2,e2,w,maxw,step,scalefact0))
    if not all(math.isfinite(v) for v in (d1,d2,e2,w,maxw,step,scale)):
        raise ValueError('scalar operands must be finite')
    if not (0<d1<=1 and d2>0 and w>0 and maxw>=0 and step>=1 and scale>1):
        raise ValueError('operands outside supported positive, growing-weight domain')

    def angle_of(value):
        norm=float(np.linalg.norm(value))
        if norm==0:
            raise ValueError('zero resultant force is outside the reconstructed domain')
        if not np.isfinite(value).all() or not math.isfinite(norm):
            raise ValueError('nonfinite resultant force is outside the reconstructed domain')
        cosine=float(np.sum(value*direction))/norm
        if not -1<=cosine<=1:
            raise ValueError('invalid acos input is outside the reconstructed domain')
        # Preserve the ELF's actual denominator rather than replacing it by math.pi.
        return math.acos(cosine)*180./3.1415926535900001

    contribution=(d1*w*d2)*direction
    force-=subtract-contribution
    angle=angle_of(force)
    updates=0
    reason='angle_satisfied'
    if angle>87.:
        while True:
            # Restore and replace in native operation order; do not accumulate
            # every attempted height as an additional Gaussian.
            force+=subtract-contribution
            next_weight=min(w*scale,w+2.)
            if not math.isfinite(next_weight) or next_weight<=w:
                raise ValueError('weight update cannot make finite representable progress')
            w=next_weight
            scale*=step
            contribution=(d1*w*d2)*direction
            force-=subtract-contribution
            angle=angle_of(force)
            updates+=1
            # Native tests the weight limit first, AFTER increasing weight.
            if w>maxw:
                reason='maxw_exceeded'
                break
            if angle<=87.:
                break
    energy=e2+d1*w
    if not math.isfinite(energy):
        raise ValueError('nonfinite output energy is outside the reconstructed domain')
    return NativeWeightResult(w,energy,force,angle,updates,reason)
