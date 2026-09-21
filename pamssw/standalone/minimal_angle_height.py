"""Analytic minimal positive height for the recovered87-degree criterion.

Independent criterion-isolation policy: no native initial weights, growth or
ceiling. No walker integration and no inference of a no-bias continuation rule.
"""
from dataclasses import dataclass
import math
import numpy as np
from .native_height_policy import FrozenHeightGaussian


@dataclass(frozen=True)
class MinimalAngleHeightResult:
    terms: tuple
    weight: float
    status: str
    angle_degrees: float
    criterion_residual: float  # computed Fparallel - cot87*norm(Fperp)
    background_parallel: float
    background_perpendicular_norm: float


class MinimalAngleHeightPolicy:
    """W=(cot87*norm(Fperp)-Fparallel)/(d1*d2), then freeze all terms.

    Forces/directions use one explicitly selected scaled-coordinate metric.
    W<=0 returns already_forward_satisfied with history unchanged, NOT a new
    zero-height term and NOT authorization to quench/advance an outer walker.
    Exact antiparallel/zero background has an unattained minimum at zero total
    force and is rejected rather than assigned an invented positive floor.
    """
    def prepare(self,history,*,center,direction,width,point,background_force):
        history=tuple(history)
        if any(not isinstance(t,FrozenHeightGaussian) for t in history):raise TypeError('frozen Gaussian history required')
        template=FrozenHeightGaussian(center,direction,width,1.)
        q=np.asarray(point,dtype=float);f=np.asarray(background_force,dtype=float)
        if q.shape!=template.center.shape or f.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(f).all():raise ValueError('compatible finite point/background force required')
        if any(t.center.shape!=q.shape for t in history):raise ValueError('history coordinate dimension mismatch')
        z=float((q-template.center)@template.direction)
        d1=math.exp(-.5*(z/template.width)**2);c=d1*z/template.width**2
        if not math.isfinite(c) or c<=0:raise ValueError('positive nonunderflowed forward Gaussian response required')
        n=template.direction;parallel=float(f@n);perp=f-parallel*n;pnorm=float(np.linalg.norm(perp))
        theta=87.*3.1415926535900001/180.;cot=1./math.tan(theta)
        target=cot*pnorm;numerator=target-parallel;weight=numerator/c
        def angle(force):
            norm=float(np.linalg.norm(force))
            if not math.isfinite(norm) or norm==0:raise ValueError('zero/nonfinite resultant has no force angle')
            cosine=float(force@n)/norm
            if not -1<=cosine<=1:raise ValueError('invalid acos operand')
            return math.acos(cosine)*180./3.1415926535900001
        if pnorm==0. and parallel<=0.:
            raise ValueError('minimal height is unattained: cancellation boundary has zero resultant')
        if numerator<=0:
            return MinimalAngleHeightResult(history,0.,'already_forward_satisfied',angle(f),parallel-target,parallel,pnorm)
        if not math.isfinite(weight) or weight<=0:raise ValueError('minimal positive height is not finite representable')
        term=FrozenHeightGaussian(template.center,n,width,weight);total=f+term.evaluate(q)[1]
        resultant_parallel=float(total@n);resultant_perp=total-resultant_parallel*n
        if resultant_parallel<=0 or not np.isfinite(total).all():raise FloatingPointError('finite arithmetic cannot resolve positive target force')
        residual=resultant_parallel-cot*float(np.linalg.norm(resultant_perp))
        return MinimalAngleHeightResult(history+(term,),weight,'prepared',angle(total),residual,parallel,pnorm)
